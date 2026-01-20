# Cluster 27

class Albumentations:

    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)
            self.transform = A.Compose([A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01), A.RandomBrightnessContrast(p=0.0), A.RandomGamma(p=0.0), A.ImageCompression(quality_lower=75, p=0.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            LOGGER.info(colorstr('albumentations: ') + ', '.join((f'{x}' for x in self.transform.transforms if x.p)))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            im, labels = (new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]))
        return (im, labels)

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    if hard:
        assert result, s
    if verbose and (not result):
        LOGGER.warning(s)
    return result

def attempt_download(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", ''))
    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')
            file = name.split('?')[0]
            if Path(file).is_file():
                print(f'Found {url} locally at {file}')
            else:
                safe_download(file=file, url=url, min_bytes=100000.0)
            return file
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']
        except:
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v6.0'
        if name in assets:
            safe_download(file, url=f'https://github.com/{repo}/releases/download/{tag}/{name}', min_bytes=100000.0, error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')
    return str(file)

def safe_download(file, url, url2=None, min_bytes=1.0, error_msg=''):
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg
    except Exception as e:
        file.unlink(missing_ok=True)
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:
            file.unlink(missing_ok=True)
            print(f'ERROR: {assert_msg}\n{error_msg}')
        print('')

class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList((nn.Conv2d(x, self.no * self.na, 1) for x in ch))
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return (grid, anchor_grid)

def run(weights=ROOT / 'yolov5s.pt', imgsz=(640, 640), batch_size=1, dynamic=False):
    im = torch.zeros((batch_size, 3, *imgsz))
    model = attempt_load(weights, map_location=torch.device('cpu'), inplace=True, fuse=False)
    y = model(im)
    model.info()
    im = tf.zeros((batch_size, *imgsz, 3))
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    y = tf_model.predict(im)
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()
    LOGGER.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')

def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()
    if len(model) == 1:
        return model[-1]
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
        return model

def main(opt):
    run(**vars(opt))

class DetectMultiBackend(nn.Module):

    def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
        from models.experimental import attempt_download, attempt_load
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix = Path(w).suffix.lower()
        suffixes = ['.pt', '.torchscript', '.onnx', '.engine', '.tflite', '.pb', '', '.mlmodel', '.xml']
        check_suffix(w, suffixes)
        pt, jit, onnx, engine, tflite, pb, saved_model, coreml, xml = (suffix == x for x in suffixes)
        stride, names = (64, [f'class{i}' for i in range(1000)])
        w = attempt_download(w)
        if data:
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']
        if pt:
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = int(model.stride.max())
            names = model.module.names if hasattr(model, 'module') else model.names
            self.model = model
        elif jit:
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])
                stride, names = (int(d['stride']), d['names'])
        elif dnn:
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))
            import openvino.inference_engine as ie
            core = ie.IECore()
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt
            check_version(trt.__version__, '7.0.0', hard=True)
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict(((n, d.ptr) for n, d in bindings.items()))
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            model = tf.keras.models.load_model(w)
        elif pb:
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=''), [])
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs), tf.nest.map_structure(x.graph.as_graph_element, outputs))
            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs='x:0', outputs='Identity:0')
        elif tflite:
            if 'edgetpu' in w.lower():
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                import tflite_runtime.interpreter as tfli
                delegate = {'Linux': 'libedgetpu.so.1', 'Darwin': 'libedgetpu.1.dylib', 'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = tfli.Interpreter(model_path=w, experimental_delegates=[tfli.load_delegate(delegate)])
            else:
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=w)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        self.__dict__.update(locals())

    def forward(self, im, augment=False, visualize=False, val=False):
        b, ch, h, w = im.shape
        if self.pt or self.jit:
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:
            im = im.cpu().numpy()
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:
            im = im.cpu().numpy()
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:
            im = im.cpu().numpy()
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')
            request = self.executable_network.requests[0]
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))
            request.infer()
            y = request.output_blobs['output'].buffer
        elif self.engine:
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:
            im = im.permute(0, 2, 3, 1).cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            y = self.model.predict({'image': im})
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])
                conf, cls = (y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float))
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = y[list(y)[-1]]
        else:
            im = im.permute(0, 2, 3, 1).cpu().numpy()
            if self.saved_model:
                y = self.model(im, training=False).numpy()
            elif self.pb:
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            elif self.tflite:
                input, output = (self.input_details[0], self.output_details[0])
                int8 = input['dtype'] == np.uint8
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale
            y[..., 0] *= w
            y[..., 1] *= h
            y[..., 2] *= w
            y[..., 3] *= h
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        if self.pt or self.jit or self.onnx or self.engine:
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)
                self.forward(im)

