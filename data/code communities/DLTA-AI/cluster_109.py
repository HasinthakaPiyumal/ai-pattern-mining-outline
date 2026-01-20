# Cluster 109

def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    try:
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')
        ts = torch.jit.trace(model, im, strict=False)
        if optimize:
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f))
        else:
            ts.save(str(f))
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')

def file_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1000000.0
    elif path.is_dir():
        return sum((f.stat().st_size for f in path.glob('**/*') if f.is_file())) / 1000000.0
    else:
        return 0.0

def export_onnx(model, im, file, opset, dynamic, fp16, simplify, prefix=colorstr('ONNX:')):
    try:
        check_requirements(('onnx',))
        import onnx
        f = file.with_suffix('.onnx')
        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        if dynamic:
            dynamic = {'images': {0: 'batch'}, 'output': {0: 'batch'}}
        torch.onnx.export(model.half() if fp16 else model.cpu(), im.half() if fp16 else im.cpu(), f, verbose=False, opset_version=opset, do_constant_folding=True, input_names=['images'], output_names=['output'], dynamic_axes=dynamic or None)
        model_onnx = onnx.load(f)
        onnx.checker.check_model(model_onnx)
        onnx.save(model_onnx, f)
        if simplify:
            try:
                cuda = torch.cuda.is_available()
                check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
                import onnxsim
                LOGGER.info(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'export failure: {e}')

def export_openvino(file, half, prefix=colorstr('OpenVINO:')):
    check_requirements(('openvino-dev',))
    import openvino.inference_engine as ie
    try:
        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        f = str(file).replace('.pt', f'_openvino_model{os.sep}')
        cmd = f'mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {('FP16' if half else 'FP32')}'
        subprocess.check_output(cmd.split())
    except Exception as e:
        LOGGER.info(f'export failure: {e}')
    LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    return f

def export_tflite(file, half, prefix=colorstr('TFLite:')):
    try:
        check_requirements(('openvino2tensorflow', 'tensorflow', 'tensorflow_datasets'))
        import openvino.inference_engine as ie
        LOGGER.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
        output = Path(str(file).replace(f'_openvino_model{os.sep}', f'_tflite_model{os.sep}'))
        modelxml = list(Path(file).glob('*.xml'))[0]
        cmd = f'openvino2tensorflow             --model_path {modelxml}             --model_output_path {output}             --output_pb             --output_saved_model             --output_no_quant_float32_tflite             --output_dynamic_range_quant_tflite'
        subprocess.check_output(cmd.split())
        LOGGER.info(f'{prefix} export success, results saved in {output} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')

def export_engine(model, im, file, half, dynamic, simplify, workspace=4, verbose=False, prefix=colorstr('TensorRT:')):
    try:
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == 'Linux':
                check_requirements(('nvidia-tensorrt',), cmds=('-U --index-url https://pypi.ngc.nvidia.com',))
            import tensorrt as trt
        if trt.__version__[0] == '7':
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, dynamic, half, simplify)
            model.model[-1].anchor_grid = grid
        else:
            check_version(trt.__version__, '8.0.0', hard=True)
            export_onnx(model, im, file, 12, dynamic, half, simplify)
        onnx = file.with_suffix('.onnx')
        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        LOGGER.info(f'{prefix} Network Description:')
        for inp in inputs:
            LOGGER.info(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
        if dynamic:
            if im.shape[0] <= 1:
                LOGGER.warning(f'{prefix}WARNING: --dynamic model requires maximum --batch-size argument')
            profile = builder.create_optimization_profile()
            for inp in inputs:
                if half:
                    inp.dtype = trt.float16
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)
        LOGGER.info(f'{prefix} building FP{(16 if builder.platform_has_fast_fp16 and half else 32)} engine in {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
            config.default_device_type = trt.DeviceType.GPU
        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'\n{prefix} export failure: {e}')

def export_formats():
    x = [['PyTorch', '-', '.pt', True, True], ['TorchScript', 'torchscript', '.torchscript', True, True], ['ONNX', 'onnx', '.onnx', True, True], ['OpenVINO', 'openvino', '_openvino_model', True, False], ['TensorRT', 'engine', '.engine', False, True], ['TensorFlow Lite', 'tflite', '.tflite', True, False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

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

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'

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

