# Cluster 17

def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod

def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath

def _traversed_config_file():
    """We traversed all potential config files under the `config` file. If you
    need to print details or debug code, you can use this function.

    If the `backbone.init_cfg` is None (do not use `Pretrained` init way), you
    need add the folder name in `ignores_folder` (if the config files in this
    folder all set backbone.init_cfg is None) or add config name in
    `ignores_file` (if the config file set backbone.init_cfg is None)
    """
    config_path = _get_config_directory()
    check_cfg_names = []
    ignores_folder = ['_base_', 'legacy_1.x', 'common']
    ignores_folder += ['ld']
    ignores_folder += ['selfsup_pretrain']
    ignores_folder += ['centripetalnet', 'cornernet', 'cityscapes', 'scratch']
    ignores_file = ['ssdlite_mobilenetv2_scratch_600e_coco.py']
    for config_file_name in os.listdir(config_path):
        if config_file_name not in ignores_folder:
            config_file = join(config_path, config_file_name)
            if os.path.isdir(config_file):
                for config_sub_file in os.listdir(config_file):
                    if config_sub_file.endswith('py') and config_sub_file not in ignores_file:
                        name = join(config_file, config_sub_file)
                        check_cfg_names.append(name)
    return check_cfg_names

@pytest.mark.parametrize('config', _traversed_config_file())
def test_load_pretrained(config):
    """Check out backbone whether successfully load pretrained model by using
    `backbone.init_cfg`.

    Details please refer to `_check_backbone`
    """
    _check_backbone(config, print_cfg=False)

def _check_backbone(config, print_cfg=True):
    """Check out backbone whether successfully load pretrained model, by using
    `backbone.init_cfg`.

    First, using `mmcv._load_checkpoint` to load the checkpoint without
        loading models.
    Then, using `build_detector` to build models, and using
        `model.init_weights()` to initialize the parameters.
    Finally, assert weights and bias of each layer loaded from pretrained
        checkpoint are equal to the weights and bias of original checkpoint.
        For the convenience of comparison, we sum up weights and bias of
        each loaded layer separately.

    Args:
        config (str): Config file path.
        print_cfg (bool): Whether print logger and return the result.

    Returns:
        results (str or None): If backbone successfully load pretrained
            checkpoint, return None; else, return config file path.
    """
    if print_cfg:
        print('-' * 15 + 'loading ', config)
    cfg = Config.fromfile(config)
    init_cfg = None
    try:
        init_cfg = cfg.model.backbone.init_cfg
        init_flag = True
    except AttributeError:
        init_flag = False
    if init_cfg is None or init_cfg.get('type') != 'Pretrained':
        init_flag = False
    if init_flag:
        checkpoint = _load_checkpoint(init_cfg.checkpoint)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        checkpoint_layers = state_dict.keys()
        for name, value in model.backbone.state_dict().items():
            if name in checkpoint_layers:
                assert value.equal(state_dict[name])
        if print_cfg:
            print('-' * 10 + 'Successfully load checkpoint' + '-' * 10 + '\n')
            return None
    elif print_cfg:
        print(config + '\n' + '-' * 10 + 'config file do not have init_cfg' + '-' * 10 + '\n')
        return config

def _test_load_pretrained():
    """We traversed all potential config files under the `config` file. If you
    need to print details or debug code, you can use this function.

    Returns:
        check_cfg_names (list[str]): Config files that backbone initialized
        from pretrained checkpoint might be problematic. Need to recheck
        the config file. The output including the config files that the
        backbone.init_cfg is None
    """
    check_cfg_names = _traversed_config_file()
    need_check_cfg = []
    prog_bar = ProgressBar(len(check_cfg_names))
    for config in check_cfg_names:
        init_cfg_name = _check_backbone(config)
        if init_cfg_name is not None:
            need_check_cfg.append(init_cfg_name)
        prog_bar.update()
    print('These config files need to be checked again')
    print(need_check_cfg)

class MMdetHandler(BaseHandler):
    threshold = 0.5

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' + str(properties.get('gpu_id')) if torch.cuda.is_available() else self.map_location)
        self.manifest = context.manifest
        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')
        self.model = init_detector(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)
        return images

    def inference(self, data, *args, **kwargs):
        results = inference_detector(self.model, data)
        return results

    def postprocess(self, data):
        output = []
        for image_index, image_result in enumerate(data):
            output.append([])
            if isinstance(image_result, tuple):
                bbox_result, segm_result = image_result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]
            else:
                bbox_result, segm_result = (image_result, None)
            for class_index, class_result in enumerate(bbox_result):
                class_name = self.model.CLASSES[class_index]
                for bbox in class_result:
                    bbox_coords = bbox[:-1].tolist()
                    score = float(bbox[-1])
                    if score >= self.threshold:
                        output[image_index].append({'class_name': class_name, 'bbox': bbox_coords, 'score': score})
        return output

def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(f'config must be a filename or Config object, but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn("Class names are not saved in the checkpoint's meta data, use COCO classes by default.")
            model.CLASSES = get_classes('coco')
    model.cfg = config
    model.to(device)
    model.eval()
    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model)
        model.cfg = config
    return model

def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
    cfg = model.cfg
    device = next(model.parameters()).device
    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            data = dict(img=img)
        else:
            data = dict(img_info=dict(filename=img), img_prefix=None)
        data = test_pipeline(data)
        datas.append(data)
    data = collate(datas, samples_per_gpu=len(imgs))
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(m, RoIPool), 'CPU inference with RoIPool is not supported currently.'
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    if not is_batch:
        return results[0]
    else:
        return results

def parse_result(input, model_class):
    bbox = []
    label = []
    score = []
    for anchor in input:
        bbox.append(anchor['bbox'])
        label.append(model_class.index(anchor['class_name']))
        score.append([anchor['score']])
    bboxes = np.append(bbox, score, axis=1)
    labels = np.array(label)
    result = bbox2result(bboxes, labels, len(model_class))
    return result

def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model_result = inference_detector(model, args.img)
    for i, anchor_set in enumerate(model_result):
        anchor_set = anchor_set[anchor_set[:, 4] >= 0.5]
        model_result[i] = anchor_set
    show_result_pyplot(model, args.img, model_result, score_thr=args.score_thr, title='pytorch_result')
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    server_result = parse_result(response.json(), model.CLASSES)
    show_result_pyplot(model, args.img, server_result, score_thr=args.score_thr, title='server_result')
    for i in range(len(model.CLASSES)):
        assert np.allclose(model_result[i], server_result[i])

def show_result_pyplot(model, img, result, score_thr=0.3, title='result', wait_time=0, palette=None, out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(img, result, score_thr=score_thr, show=True, wait_time=wait_time, win_name=title, bbox_color=palette, text_color=(200, 200, 200), mask_color=palette, out_file=out_file)

def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod

def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model

def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod

def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model

def test_inference_detector():
    from mmcv import ConfigDict
    from mmdet.apis import inference_detector
    from mmdet.models import build_detector
    num_class = 3
    model_dict = dict(type='RetinaNet', backbone=dict(type='ResNet', depth=18, num_stages=4, out_indices=(3,), norm_cfg=dict(type='BN', requires_grad=False), norm_eval=True, style='pytorch'), neck=None, bbox_head=dict(type='RetinaHead', num_classes=num_class, in_channels=512, stacked_convs=1, feat_channels=256, anchor_generator=dict(type='AnchorGenerator', octave_base_scale=4, scales_per_octave=3, ratios=[0.5], strides=[32]), bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0])), test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
    rng = np.random.RandomState(0)
    img1 = rng.rand(100, 100, 3)
    img2 = rng.rand(100, 100, 3)
    model = build_detector(ConfigDict(model_dict))
    config = _get_config_module('retinanet/retinanet_r50_fpn_1x_coco.py')
    model.cfg = config
    result = inference_detector(model, img1)
    assert len(result) == num_class
    result = inference_detector(model, [img1, img2])
    assert len(result) == 2 and len(result[0]) == num_class

@contextlib.contextmanager
def profile_time(trace_name, name, enabled=True, stream=None, end_stream=None):
    """Print time spent by CPU and GPU.

        Useful as a temporary context manager to find sweet spots of code
        suitable for async implementation.
        """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    stream = stream if stream else torch.cuda.current_stream()
    end_stream = end_stream if end_stream else stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        cpu_start = time.monotonic()
        yield
    finally:
        cpu_end = time.monotonic()
        end_stream.record_event(end)
        end.synchronize()
        cpu_time = (cpu_end - cpu_start) * 1000
        gpu_time = start.elapsed_time(end)
        msg = f'{trace_name} {name} cpu_time {cpu_time:.2f} ms '
        msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
        print(msg, end_stream)

class MaskRCNNDetector:

    def __init__(self, model_config, checkpoint=None, streamqueue_size=3, device='cuda:0'):
        self.streamqueue_size = streamqueue_size
        self.device = device
        self.model = init_detector(model_config, checkpoint=None, device=self.device)
        self.streamqueue = None

    async def init(self):
        self.streamqueue = asyncio.Queue()
        for _ in range(self.streamqueue_size):
            stream = torch.cuda.Stream(device=self.device)
            self.streamqueue.put_nowait(stream)
    if sys.version_info >= (3, 7):

        async def apredict(self, img):
            if isinstance(img, str):
                img = mmcv.imread(img)
            async with concurrent(self.streamqueue):
                result = await async_inference_detector(self.model, img)
            return result

def test_init_detector():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, '..')
    config_file = os.path.join(project_dir, 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')
    cfg_options = dict(model=dict(backbone=dict(depth=18, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'))))
    model = init_detector(config_file, device='cpu', cfg_options=cfg_options)
    config_path_object = Path(config_file)
    model = init_detector(config_path_object, device='cpu')
    with pytest.raises(TypeError):
        config_list = [config_file]
        model = init_detector(config_list)

@pytest.mark.parametrize('config_rpath', ['wider_face/ssd300_wider_face.py', 'pascal_voc/ssd300_voc0712.py', 'pascal_voc/ssd512_voc0712.py', 'foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py', 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py', 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain_1x_coco.py', 'mask_rcnn/mask_rcnn_r50_fpn_fp16_1x_coco.py'])
def test_config_data_pipeline(config_rpath):
    """Test whether the data pipeline is valid and can process corner cases.

    CommandLine:
        xdoctest -m tests/test_runtime/
            test_config.py test_config_build_data_pipeline
    """
    import numpy as np
    from mmcv import Config
    from mmdet.datasets.pipelines import Compose
    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath}')

    def dummy_masks(h, w, num_obj=3, mode='bitmap'):
        assert mode in ('polygon', 'bitmap')
        if mode == 'bitmap':
            masks = np.random.randint(0, 2, (num_obj, h, w), dtype=np.uint8)
            masks = BitmapMasks(masks, h, w)
        else:
            masks = []
            for i in range(num_obj):
                masks.append([])
                masks[-1].append(np.random.uniform(0, min(h - 1, w - 1), (8 + 4 * i,)))
                masks[-1].append(np.random.uniform(0, min(h - 1, w - 1), (10 + 4 * i,)))
            masks = PolygonMasks(masks, h, w)
        return masks
    config_fpath = join(config_dpath, config_rpath)
    cfg = Config.fromfile(config_fpath)
    loading_pipeline = cfg.train_pipeline.pop(0)
    loading_ann_pipeline = cfg.train_pipeline.pop(0)
    cfg.test_pipeline.pop(0)
    train_pipeline = Compose(cfg.train_pipeline)
    test_pipeline = Compose(cfg.test_pipeline)
    print(f'Building data pipeline, config_fpath = {config_fpath}')
    print(f'Test training data pipeline: \n{train_pipeline!r}')
    img = np.random.randint(0, 255, size=(888, 666, 3), dtype=np.uint8)
    if loading_pipeline.get('to_float32', False):
        img = img.astype(np.float32)
    mode = 'bitmap' if loading_ann_pipeline.get('poly2mask', True) else 'polygon'
    results = dict(filename='test_img.png', ori_filename='test_img.png', img=img, img_shape=img.shape, ori_shape=img.shape, gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32), gt_labels=np.array([1], dtype=np.int64), gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode))
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = train_pipeline(results)
    assert output_results is not None
    print(f'Test testing data pipeline: \n{test_pipeline!r}')
    results = dict(filename='test_img.png', ori_filename='test_img.png', img=img, img_shape=img.shape, ori_shape=img.shape, gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32), gt_labels=np.array([1], dtype=np.int64), gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode))
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = test_pipeline(results)
    assert output_results is not None
    print(f'Test empty GT with training data pipeline: \n{train_pipeline!r}')
    results = dict(filename='test_img.png', ori_filename='test_img.png', img=img, img_shape=img.shape, ori_shape=img.shape, gt_bboxes=np.zeros((0, 4), dtype=np.float32), gt_labels=np.array([], dtype=np.int64), gt_masks=dummy_masks(img.shape[0], img.shape[1], num_obj=0, mode=mode))
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = train_pipeline(results)
    assert output_results is not None
    print(f'Test empty GT with testing data pipeline: \n{test_pipeline!r}')
    results = dict(filename='test_img.png', ori_filename='test_img.png', img=img, img_shape=img.shape, ori_shape=img.shape, gt_bboxes=np.zeros((0, 4), dtype=np.float32), gt_labels=np.array([], dtype=np.int64), gt_masks=dummy_masks(img.shape[0], img.shape[1], num_obj=0, mode=mode))
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = test_pipeline(results)
    assert output_results is not None

def dummy_masks(h, w, num_obj=3, mode='bitmap'):
    assert mode in ('polygon', 'bitmap')
    if mode == 'bitmap':
        masks = np.random.randint(0, 2, (num_obj, h, w), dtype=np.uint8)
        masks = BitmapMasks(masks, h, w)
    else:
        masks = []
        for i in range(num_obj):
            masks.append([])
            masks[-1].append(np.random.uniform(0, min(h - 1, w - 1), (8 + 4 * i,)))
            masks[-1].append(np.random.uniform(0, min(h - 1, w - 1), (10 + 4 * i,)))
        masks = PolygonMasks(masks, h, w)
    return masks

