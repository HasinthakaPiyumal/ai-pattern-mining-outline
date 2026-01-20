# Cluster 120

class ReIDDetectMultiBackend(nn.Module):

    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        w = weights[0] if isinstance(weights, list) else weights
        self.pt, self.jit, self.onnx, self.xml, self.engine, self.tflite = self.model_type(w)
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine
        self.device = device
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()
        model_name = get_model_name(w)
        if w.suffix == '.pt':
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                print(f'No URL associated to the chosen StrongSORT weights ({w}). Choose between:')
                show_downloadeable_models()
                exit()
        self.model = build_model(model_name, num_classes=1, pretrained=not (w and w.is_file()), use_gpu=device)
        if self.pt:
            if w and w.is_file() and (w.suffix == '.pt'):
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.jit:
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt
            check_version(trt.__version__, '7.0.0', hard=True)
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict(((n, d.ptr) for n, d in self.bindings.items()))
            batch_size = self.bindings['images'].shape[0]
        elif self.xml:
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino',))
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():
                w = next(Path(w).glob('*.xml'))
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCWH'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            self.executable_network = ie.compile_model(network, device_name='CPU')
            self.output_layer = next(iter(self.executable_network.outputs))
        elif self.tflite:
            LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = (tf.lite.Interpreter, tf.lite.experimental.load_delegate)
            self.interpreter = tf.lite.Interpreter(model_path=w)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            input_data = np.array(np.random.random_sample((1, 256, 128, 3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            print('This model framework is not supported yet!')
            exit()

    @staticmethod
    def model_type(p='path/to/model.pt'):
        from trackers.reid_export import export_formats
        sf = list(export_formats().Suffix)
        check_suffix(p, sf)
        types = [s in Path(p).name for s in sf]
        return types

    def _preprocess(self, im_batch):
        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(self.device)
        return images

    def forward(self, im_batch):
        im_batch = self._preprocess(im_batch)
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:
            features = self.model(im_batch)
        elif self.onnx:
            im_batch = im_batch.cpu().numpy()
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f'input size {im_batch.shape} {('>' if self.dynamic else 'not equal to')} max model size {s}'
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        elif self.xml:
            im_batch = im_batch.cpu().numpy()
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            print('Framework not supported at the moment, we are working on it...')
            exit()
        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        warmup_types = (self.pt, self.jit, self.onnx, self.engine, self.tflite)
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]
            for _ in range(2 if self.jit else 1):
                self.forward(im)

def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None

def get_model_url(model):
    if model.name in __trained_urls:
        return __trained_urls[model.name]
    else:
        None

def show_downloadeable_models():
    print('\nAvailable .pt ReID models for automatic download')
    print(list(__trained_urls.keys()))

def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu)

def load_pretrained_weights(model, weight_path):
    """Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = ([], [])
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded due to unmatched keys or layer size: {}'.format(discarded_layers))

class ReIDDetectMultiBackend(nn.Module):

    def __init__(self, weights='osnet_x0_25_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()
        w = weights[0] if isinstance(weights, list) else weights
        self.pt, self.jit, self.onnx, self.xml, self.engine, self.tflite = self.model_type(w)
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine
        self.device = device
        self.image_size = (256, 128)
        self.pixel_mean = [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()
        model_name = get_model_name(w)
        if w.suffix == '.pt':
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                print(f'No URL associated to the chosen StrongSORT weights ({w}). Choose between:')
                show_downloadeable_models()
                exit()
        self.model = build_model(model_name, num_classes=1, pretrained=not (w and w.is_file()), use_gpu=device)
        if self.pt:
            if w and w.is_file() and (w.suffix == '.pt'):
                load_pretrained_weights(self.model, w)
            self.model.to(device).eval()
            self.model.half() if self.fp16 else self.model.float()
        elif self.jit:
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            self.model = torch.jit.load(w)
            self.model.half() if self.fp16 else self.model.float()
        elif self.onnx:
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available() and device.type != 'cpu'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)
        elif self.engine:
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt
            check_version(trt.__version__, '7.0.0', hard=True)
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model_ = runtime.deserialize_cuda_engine(f.read())
            self.context = self.model_.create_execution_context()
            self.bindings = OrderedDict()
            self.fp16 = False
            dynamic = False
            for index in range(self.model_.num_bindings):
                name = self.model_.get_binding_name(index)
                dtype = trt.nptype(self.model_.get_binding_dtype(index))
                if self.model_.binding_is_input(index):
                    if -1 in tuple(self.model_.get_binding_shape(index)):
                        dynamic = True
                        self.context.set_binding_shape(index, tuple(self.model_.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                shape = tuple(self.context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict(((n, d.ptr) for n, d in self.bindings.items()))
            batch_size = self.bindings['images'].shape[0]
        elif self.xml:
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino',))
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():
                w = next(Path(w).glob('*.xml'))
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCWH'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            self.executable_network = ie.compile_model(network, device_name='CPU')
            self.output_layer = next(iter(self.executable_network.outputs))
        elif self.tflite:
            LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = (tf.lite.Interpreter, tf.lite.experimental.load_delegate)
            self.interpreter = tf.lite.Interpreter(model_path=w)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            input_data = np.array(np.random.random_sample((1, 256, 128, 3)), dtype=np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            print('This model framework is not supported yet!')
            exit()

    @staticmethod
    def model_type(p='path/to/model.pt'):
        from trackers.reid_export import export_formats
        sf = list(export_formats().Suffix)
        check_suffix(p, sf)
        types = [s in Path(p).name for s in sf]
        return types

    def _preprocess(self, im_batch):
        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.to(self.device)
        return images

    def forward(self, im_batch):
        im_batch = self._preprocess(im_batch)
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:
            features = self.model(im_batch)
        elif self.onnx:
            im_batch = im_batch.cpu().numpy()
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f'input size {im_batch.shape} {('>' if self.dynamic else 'not equal to')} max model size {s}'
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        elif self.xml:
            im_batch = im_batch.cpu().numpy()
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            print('Framework not supported at the moment, we are working on it...')
            exit()
        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        warmup_types = (self.pt, self.jit, self.onnx, self.engine, self.tflite)
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]
            for _ in range(2 if self.jit else 1):
                self.forward(im)

