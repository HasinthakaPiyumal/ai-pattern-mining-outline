# Cluster 28

def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]
        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))
        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha
    p = PatchCollection(polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)
    return (ax, img)

def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)

def imshow_det_bboxes(img, bboxes=None, labels=None, segms=None, class_names=None, score_thr=0, bbox_color='green', text_color='green', mask_color=None, thickness=2, font_size=8, win_name='', show=True, wait_time=0, out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
           If a single color is given, it will be applied to all classes.
           The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
           masks. If a single color is given, it will be applied to all
           classes. The tuple of color should be in RGB order.
           Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], 'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], 'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, 'segms and bboxes should not be None at the same time.'
    img = mmcv.imread(img).astype(np.uint8)
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    img = mmcv.bgr2rgb(img)
    width, height = (img.shape[1], img.shape[0])
    img = np.ascontiguousarray(img)
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]
    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)
        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(ax, labels[:num_bboxes], positions, scores=scores, class_names=class_names, color=text_colors, font_size=font_size, scales=scales, horizontal_alignment=horizontal_alignment)
    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)
        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(ax, labels[num_bboxes:], positions, class_names=class_names, color=text_colors, font_size=font_size, scales=scales, horizontal_alignment=horizontal_alignment)
    plt.imshow(img)
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)
    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    plt.close()
    return img

def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]], [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=thickness, alpha=alpha)
    ax.add_collection(p)
    return ax

def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales

def draw_labels(ax, labels, positions, scores=None, class_names=None, color='w', font_size=8, scales=None, horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[label] if class_names is not None else f'class {label}'
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color
        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(pos[0], pos[1], f'{label_text}', bbox={'facecolor': 'black', 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}, color=text_color, fontsize=font_size_mask, verticalalignment='top', horizontalalignment=horizontal_alignment)
    return ax

def imshow_gt_det_bboxes(img, annotation, result, class_names=None, score_thr=0, gt_bbox_color=(61, 102, 255), gt_text_color=(200, 200, 200), gt_mask_color=(61, 102, 255), det_bbox_color=(241, 101, 72), det_text_color=(200, 200, 200), det_mask_color=(241, 101, 72), thickness=2, font_size=13, win_name='', show=True, wait_time=0, out_file=None, overlay_gt_pred=True):
    """General visualization GT and result function.

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown. Default: 0.
      gt_bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      gt_mask_color (list[tuple] | tuple | str | None, optional): Colors of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      det_bbox_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_mask_color (list[tuple] | tuple | str | None, optional): Color of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.
      overlay_gt_pred (bool): Whether to plot gts and predictions on the
       same image. If False, predictions and gts will be plotted on two same
       image which will be concatenated in vertical direction. The image
       above is drawn with gt, and the image below is drawn with the
       prediction result. Default: True.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), f'Expected tuple or list or dict, but get {type(result)}'
    gt_bboxes = annotation['gt_bboxes']
    gt_labels = annotation['gt_labels']
    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)
    gt_seg = annotation.get('gt_semantic_seg', None)
    if gt_seg is not None:
        pad_value = 255
        sem_labels = np.unique(gt_seg)
        all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
        all_labels, counts = np.unique(all_labels, return_counts=True)
        stuff_labels = all_labels[np.logical_and(counts < 2, all_labels != pad_value)]
        stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
        gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
        gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)), axis=0)
    img = mmcv.imread(img)
    img_with_gt = imshow_det_bboxes(img, gt_bboxes, gt_labels, gt_masks, class_names=class_names, bbox_color=gt_bbox_color, text_color=gt_text_color, mask_color=gt_mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=False)
    if not isinstance(result, dict):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = (result, None)
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        segms = None
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
    else:
        assert class_names is not None, 'We need to know the number of classes.'
        VOID = len(class_names)
        bboxes = None
        pan_results = result['pan_results']
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != VOID
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = pan_results[None] == ids[:, None, None]
    if overlay_gt_pred:
        img = imshow_det_bboxes(img_with_gt, bboxes, labels, segms=segms, class_names=class_names, score_thr=score_thr, bbox_color=det_bbox_color, text_color=det_text_color, mask_color=det_mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
    else:
        img_with_det = imshow_det_bboxes(img, bboxes, labels, segms=segms, class_names=class_names, score_thr=score_thr, bbox_color=det_bbox_color, text_color=det_text_color, mask_color=det_mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=False)
        img = np.concatenate([img_with_gt, img_with_det], axis=0)
        plt.imshow(img)
        if show:
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        plt.close()
    return img

class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_bbox or (hasattr(self, 'bbox_head') and self.bbox_head is not None)

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_mask or (hasattr(self, 'mask_head') and self.mask_head is not None)

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) != num of image metas ({len(img_metas)})')
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1
        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})')
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, f'aug test does not support inference with batch size {imgs[0].size(0)}'
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor                 which may be a weighted sum of all losses, log_vars contains                 all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum((_loss.mean() for _loss in loss_value))
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum((_value for _key, _value in log_vars.items() if 'loss' in _key))
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' + ','.join(log_vars.keys())
            assert log_var_length == len(log_vars) * dist.get_world_size(), 'loss log variables are different across GPUs!\n' + message
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return (loss, log_vars)

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,                 ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def show_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]
        else:
            bbox_result, segm_result = (result, None)
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        segms = None
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, bboxes, labels, segms, class_names=self.CLASSES, score_thr=score_thr, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does not support ONNX EXPORT')

@DETECTORS.register_module()
class MaskFormer(SingleStageDetector):
    """Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""

    def __init__(self, backbone, neck=None, panoptic_head=None, panoptic_fusion_head=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)
        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)
        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.num_stuff_classes > 0:
            self.show_result = self._show_pan_result

    def forward_dummy(self, img, img_metas):
        """Used for computing network flops. See
        `mmdetection/tools/analysis_tools/get_flops.py`

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        outs = self.panoptic_head(x, img_metas)
        return outs

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg=None, gt_bboxes_ignore=None, **kargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.panoptic_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_masks, gt_semantic_seg, gt_bboxes_ignore)
        return losses

    def simple_test(self, imgs, img_metas, **kwargs):
        """Test without augmentation.

        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.

        Returns:
            list[dict[str, np.array | tuple[list]] | tuple[list]]:
                Semantic segmentation results and panoptic segmentation                 results of each image for panoptic segmentation, or formatted                 bbox and mask results of each image for instance segmentation.

            .. code-block:: none

                [
                    # panoptic segmentation
                    {
                        'pan_results': np.array, # shape = [h, w]
                        'ins_results': tuple[list],
                        # semantic segmentation results are not supported yet
                        'sem_results': np.array
                    },
                    ...
                ]

            or

            .. code-block:: none

                [
                    # instance segmentation
                    (
                        bboxes, # list[np.array]
                        masks # list[list[np.array]]
                    ),
                    ...
                ]
        """
        feats = self.extract_feat(imgs)
        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)
        for i in range(len(results)):
            if 'pan_results' in results[i]:
                results[i]['pan_results'] = results[i]['pan_results'].detach().cpu().numpy()
            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i]['ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image, self.num_things_classes)
                mask_results = [[] for _ in range(self.num_things_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = (bbox_results, mask_results)
            assert 'sem_results' not in results[i], 'segmantic segmentation results are not supported yet.'
        if self.num_stuff_classes == 0:
            results = [res['ins_results'] for res in results]
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    def _show_pan_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `panoptic result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = pan_results[None] == ids[:, None, None]
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, segms=segms, labels=labels, class_names=self.CLASSES, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

@DETECTORS.register_module()
class TwoStagePanopticSegmentor(TwoStageDetector):
    """Base class of Two-stage Panoptic Segmentor.

    As well as the components in TwoStageDetector, Panoptic Segmentor has extra
    semantic_head and panoptic_fusion_head.
    """

    def __init__(self, backbone, neck=None, rpn_head=None, roi_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None, semantic_head=None, panoptic_fusion_head=None):
        super(TwoStagePanopticSegmentor, self).__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg, pretrained, init_cfg)
        if semantic_head is not None:
            self.semantic_head = build_head(semantic_head)
        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = build_head(panoptic_fusion_head_)
            self.num_things_classes = self.panoptic_fusion_head.num_things_classes
            self.num_stuff_classes = self.panoptic_fusion_head.num_stuff_classes
            self.num_classes = self.panoptic_fusion_head.num_classes

    @property
    def with_semantic_head(self):
        return hasattr(self, 'semantic_head') and self.semantic_head is not None

    @property
    def with_panoptic_fusion_head(self):
        return hasattr(self, 'panoptic_fusion_heads') and self.panoptic_fusion_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        raise NotImplementedError(f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, gt_semantic_seg=None, proposals=None, **kwargs):
        x = self.extract_feat(img)
        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        semantic_loss = self.semantic_head.forward_train(x, gt_semantic_seg)
        losses.update(semantic_loss)
        return losses

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""
        img_shapes = tuple((meta['ori_shape'] for meta in img_metas)) if rescale else tuple((meta['pad_shape'] for meta in img_metas))
        scale_factors = tuple((meta['scale_factor'] for meta in img_metas))
        if all((det_bbox.shape[0] == 0 for det_bbox in det_bboxes)):
            masks = []
            for img_shape in img_shapes:
                out_shape = (0, self.roi_head.bbox_head.num_classes) + img_shape[:2]
                masks.append(det_bboxes[0].new_zeros(out_shape))
            mask_pred = det_bboxes[0].new_zeros((0, 80, 28, 28))
            mask_results = dict(masks=masks, mask_pred=mask_pred, mask_feats=None)
            return mask_results
        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = [det_bboxes[0].new_tensor(scale_factor) for scale_factor in scale_factors]
            _bboxes = [_bboxes[i] * scale_factors[i] for i in range(len(_bboxes))]
        mask_rois = bbox2roi(_bboxes)
        mask_results = self.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)
        masks = []
        for i in range(len(_bboxes)):
            det_bbox = det_bboxes[i][:, :4]
            det_label = det_labels[i]
            mask_pred = mask_preds[i].sigmoid()
            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, det_label][:, None]
            img_h, img_w, _ = img_shapes[i]
            mask_pred, _ = _do_paste_mask(mask_pred, det_bbox, img_h, img_w, skip_empty=False)
            masks.append(mask_pred)
        mask_results['masks'] = masks
        return mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without Augmentation."""
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        bboxes, scores = self.roi_head.simple_test_bboxes(x, img_metas, proposal_list, None, rescale=rescale)
        pan_cfg = self.test_cfg.panoptic
        det_bboxes = []
        det_labels = []
        for bboxe, score in zip(bboxes, scores):
            det_bbox, det_label = multiclass_nms(bboxe, score, pan_cfg.score_thr, pan_cfg.nms, pan_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        mask_results = self.simple_test_mask(x, img_metas, det_bboxes, det_labels, rescale=rescale)
        masks = mask_results['masks']
        seg_preds = self.semantic_head.simple_test(x, img_metas, rescale)
        results = []
        for i in range(len(det_bboxes)):
            pan_results = self.panoptic_fusion_head.simple_test(det_bboxes[i], det_labels[i], masks[i], seg_preds[i])
            pan_results = pan_results.int().detach().cpu().numpy()
            result = dict(pan_results=pan_results)
            results.append(result)
        return results

    def show_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        pan_results = result['pan_results']
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != self.num_classes
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = pan_results[None] == ids[:, None, None]
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, segms=segms, labels=labels, class_names=self.CLASSES, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

@DETECTORS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self, backbone, neck=None, bbox_head=None, mask_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            backbone.pretrained = pretrained
        super(SingleStageInstanceSegmentor, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = build_head(bbox_head)
        else:
            self.bbox_head = None
        assert mask_head, f'`mask_head` must be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = build_head(mask_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone and neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError(f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self, img, img_metas, gt_masks, gt_labels, gt_bboxes=None, gt_bboxes_ignore=None, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (B, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_masks (list[:obj:`BitmapMasks`] | None) : The segmentation
                masks for each box.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes (list[Tensor]): Each item is the truth boxes
                of each image in [tl_x, tl_y, br_x, br_y] format.
                Default: None.
            gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gt_masks = [gt_mask.to_tensor(dtype=torch.bool, device=img.device) for gt_mask in gt_masks]
        x = self.extract_feat(img)
        losses = dict()
        if self.bbox_head:
            bbox_head_preds = self.bbox_head(x)
            det_losses, positive_infos = self.bbox_head.loss(*bbox_head_preds, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks, img_metas=img_metas, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None
        mask_loss = self.mask_head.forward_train(x, gt_labels, gt_masks, img_metas, positive_infos=positive_infos, gt_bboxes=gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, **kwargs)
        assert not set(mask_loss.keys()) & set(losses.keys())
        losses.update(mask_loss)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (B, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list(tuple): Formatted bbox and mask results of multiple                 images. The outer list corresponds to each image.                 Each tuple contains two type of results of single image:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        feat = self.extract_feat(img)
        if self.bbox_head:
            outs = self.bbox_head(feat)
            results_list = self.bbox_head.get_results(*outs, img_metas=img_metas, cfg=self.test_cfg, rescale=rescale)
        else:
            results_list = None
        results_list = self.mask_head.simple_test(feat, img_metas, rescale=rescale, instances_list=results_list)
        format_results_list = []
        for results in results_list:
            format_results_list.append(self.format_results(results))
        return format_results_list

    def format_results(self, results):
        """Format the model predictions according to the interface with
        dataset.

        Args:
            results (:obj:`InstanceData`): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        data_keys = results.keys()
        assert 'scores' in data_keys
        assert 'labels' in data_keys
        assert 'masks' in data_keys, 'results should contain masks when format the results '
        mask_results = [[] for _ in range(self.mask_head.num_classes)]
        num_masks = len(results)
        if num_masks == 0:
            bbox_results = [np.zeros((0, 5), dtype=np.float32) for _ in range(self.mask_head.num_classes)]
            return (bbox_results, mask_results)
        labels = results.labels.detach().cpu().numpy()
        if 'bboxes' not in results:
            results.bboxes = results.scores.new_zeros(len(results), 4)
        det_bboxes = torch.cat([results.bboxes, results.scores[:, None]], dim=-1)
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [det_bboxes[labels == i, :] for i in range(self.mask_head.num_classes)]
        masks = results.masks.detach().cpu().numpy()
        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)
        return (bbox_results, mask_results)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def show_result(self, img, result, score_thr=0.3, bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None, thickness=2, font_size=13, win_name='', show=False, wait_time=0, out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (tuple): Format bbox and mask results.
                It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.

            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        assert isinstance(result, tuple)
        bbox_result, mask_result = result
        bboxes = np.vstack(bbox_result)
        img = mmcv.imread(img)
        img = img.copy()
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        if len(labels) == 0:
            bboxes = np.zeros([0, 5])
            masks = np.zeros([0, 0, 0])
        else:
            masks = mmcv.concat_list(mask_result)
            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(masks, axis=0)
            if bboxes[:, :4].sum() == 0:
                num_masks = len(bboxes)
                x_any = masks.any(axis=1)
                y_any = masks.any(axis=2)
                for idx in range(num_masks):
                    x = np.where(x_any[idx, :])[0]
                    y = np.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        bboxes[idx, :4] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
        if out_file is not None:
            show = False
        img = imshow_det_bboxes(img, bboxes, labels, masks, class_names=self.CLASSES, score_thr=score_thr, bbox_color=bbox_color, text_color=text_color, mask_color=mask_color, thickness=thickness, font_size=font_size, win_name=win_name, show=show, wait_time=wait_time, out_file=out_file)
        if not (show or out_file):
            return img

def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [p for p in cfg.data.train.pipeline if p['type'] != 'SegRescale']
    dataset = build_dataset(cfg.data.train)
    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        filename = os.path.join(args.output_dir, Path(item['filename']).name) if args.output_dir is not None else None
        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']
        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)
        gt_seg = item.get('gt_semantic_seg', None)
        if gt_seg is not None:
            pad_value = 255
            sem_labels = np.unique(gt_seg)
            all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
            all_labels, counts = np.unique(all_labels, return_counts=True)
            stuff_labels = all_labels[np.logical_and(counts < 2, all_labels != pad_value)]
            stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
            gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
            gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)), axis=0)
            gt_bboxes = None
        imshow_det_bboxes(item['img'], gt_bboxes, gt_labels, gt_masks, class_names=dataset.CLASSES, show=not args.not_show, wait_time=args.show_interval, out_file=filename, bbox_color=dataset.PALETTE, text_color=(200, 200, 200), mask_color=dataset.PALETTE)
        progress_bar.update()

def mask2ndarray(mask):
    """Convert Mask to ndarray..

    Args:
        mask (:obj:`BitmapMasks` or :obj:`PolygonMasks` or
        torch.Tensor or np.ndarray): The mask to be converted.

    Returns:
        np.ndarray: Ndarray mask of shape (n, h, w) that has been converted
    """
    if isinstance(mask, (BitmapMasks, PolygonMasks)):
        mask = mask.to_ndarray()
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        raise TypeError(f'Unsupported {type(mask)} data type')
    return mask

class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True.
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0.
        overlay_gt_pred (bool): Whether to plot gts and predictions on the
            same image. If False, predictions and gts will be plotted on two
            same image which will be concatenated in vertical direction.
            The image above is drawn with gt, and the image below is drawn
            with the prediction result. Default: False.
    """

    def __init__(self, show=False, wait_time=0, score_thr=0, overlay_gt_pred=False):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.overlay_gt_pred = overlay_gt_pred

    def _save_image_gts_results(self, dataset, results, performances, out_dir=None):
        """Display or save image with groung truths and predictions from a
        model.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            performances (dict): A dict contains samples's indices
                in dataset and model's performance on them.
            out_dir (str, optional): The filename to write the image.
                Defaults: None.
        """
        mmcv.mkdir_or_exist(out_dir)
        for performance_info in performances:
            index, performance = performance_info
            data_info = dataset.prepare_train_img(index)
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(performance, 3)) + name
            out_file = osp.join(out_dir, save_filename)
            imshow_gt_det_bboxes(data_info['img'], data_info, results[index], dataset.CLASSES, gt_bbox_color=dataset.PALETTE, gt_text_color=(200, 200, 200), gt_mask_color=dataset.PALETTE, det_bbox_color=dataset.PALETTE, det_text_color=(200, 200, 200), det_mask_color=dataset.PALETTE, show=self.show, score_thr=self.score_thr, wait_time=self.wait_time, out_file=out_file, overlay_gt_pred=self.overlay_gt_pred)

    def evaluate_and_show(self, dataset, results, topk=20, show_dir='work_dir'):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None.
        """
        assert topk > 0
        if topk * 2 > len(dataset):
            topk = len(dataset) // 2
        if isinstance(results[0], dict):
            good_samples, bad_samples = self.panoptic_evaluate(dataset, results, topk=topk)
        elif isinstance(results[0], list):
            good_samples, bad_samples = self.detection_evaluate(dataset, results, topk=topk)
        elif isinstance(results[0], tuple):
            results_ = [result[0] for result in results]
            good_samples, bad_samples = self.detection_evaluate(dataset, results_, topk=topk)
        else:
            raise 'The format of result is not supported yet. Current dict for panoptic segmentation and list or tuple for object detection are supported.'
        good_dir = osp.abspath(osp.join(show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(show_dir, 'bad'))
        self._save_image_gts_results(dataset, results, good_samples, good_dir)
        self._save_image_gts_results(dataset, results, bad_samples, bad_dir)

    def detection_evaluate(self, dataset, results, topk=20, eval_fn=None):
        """Evaluation for object detection.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            eval_fn (callable, optional): Eval function, Default: None.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_mAPs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_mAPs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """
        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)
        prog_bar = mmcv.ProgressBar(len(results))
        _mAPs = {}
        for i, (result,) in enumerate(zip(results)):
            data_info = dataset.prepare_train_img(i)
            mAP = eval_fn(result, data_info['ann_info'])
            _mAPs[i] = mAP
            prog_bar.update()
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        good_mAPs = _mAPs[-topk:]
        bad_mAPs = _mAPs[:topk]
        return (good_mAPs, bad_mAPs)

    def panoptic_evaluate(self, dataset, results, topk=20):
        """Evaluation for panoptic segmentation.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Panoptic segmentation results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_pqs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_pqs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """
        gt_json = dataset.coco.img_ann_map
        result_files, tmp_dir = dataset.format_results(results)
        pred_json = mmcv.load(result_files['panoptic'])['annotations']
        pred_folder = osp.join(tmp_dir.name, 'panoptic')
        gt_folder = dataset.seg_prefix
        pqs = {}
        prog_bar = mmcv.ProgressBar(len(results))
        for i in range(len(results)):
            data_info = dataset.prepare_train_img(i)
            image_id = data_info['img_info']['id']
            gt_ann = {'image_id': image_id, 'segments_info': gt_json[image_id], 'file_name': data_info['img_info']['segm_file']}
            pred_ann = pred_json[i]
            pq_stat = pq_compute_single_core(i, [(gt_ann, pred_ann)], gt_folder, pred_folder, dataset.categories, dataset.file_client, print_log=False)
            pq_results, classwise_results = pq_stat.pq_average(dataset.categories, isthing=None)
            pqs[i] = pq_results['pq']
            prog_bar.update()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        pqs = list(sorted(pqs.items(), key=lambda kv: kv[1]))
        good_pqs = pqs[-topk:]
        bad_pqs = pqs[:topk]
        return (good_pqs, bad_pqs)

def test_mask2ndarray():
    raw_masks = np.ones((3, 28, 28))
    bitmap_mask = BitmapMasks(raw_masks, 28, 28)
    output_mask = mask2ndarray(bitmap_mask)
    assert np.allclose(raw_masks, output_mask)
    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    output_mask = mask2ndarray(polygon_masks)
    assert output_mask.shape == (3, 28, 28)
    raw_masks = np.ones((3, 28, 28))
    output_mask = mask2ndarray(raw_masks)
    assert np.allclose(raw_masks, output_mask)
    raw_masks = torch.ones((3, 28, 28))
    output_mask = mask2ndarray(raw_masks)
    assert np.allclose(raw_masks, output_mask)
    raw_masks = []
    with pytest.raises(TypeError):
        output_mask = mask2ndarray(raw_masks)

