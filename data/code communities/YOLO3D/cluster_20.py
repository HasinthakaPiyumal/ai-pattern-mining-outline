# Cluster 20

def output_to_target(output):
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs ** 0.5)
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        if i == max_subplots:
            break
        x, y = (int(w * (i // ns)), int(h * (i % ns)))
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple((int(x * ns) for x in (w, h))))
    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = (int(w * (i // ns)), int(h * (i % ns)))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6
            conf = None if labels else ti[:, 6]
            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= w
                    boxes[[1, 3]] *= h
                elif scale < 1:
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def plot_val_txt():
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = (box[:, 0], box[:, 1])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)

@try_except
@Timeout(30)
def plot_labels(labels, names=(), save_dir=Path('')):
    LOGGER.info(f'Plotting labels to {save_dir / 'labels.jpg'}... ')
    c, b = (labels[:, 0], labels[:, 1:].transpose())
    nc = int(c.max() + 1)
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()
    matplotlib.use('svg')
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)
    labels[:, 1:3] = 0.5
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))
    ax[1].imshow(img)
    ax[1].axis('off')
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)
    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

def verify_image_label(args):
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = (0, 0, 0, 0, '', [])
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:
                    l = l[i]
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            l = np.zeros((0, 5), dtype=np.float32)
        return (im_file, l, shape, segments, nm, nf, ne, nc, msg)
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]

def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass
    return s

def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))

def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).long()
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= 255
                ims.append(im)
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][pred_cls1 == pred_cls2]
    return x

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

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

