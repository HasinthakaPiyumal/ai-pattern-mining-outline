# Cluster 5

@torch.no_grad()
def detect2d(weights, source, data, imgsz, device, classes):
    bbox_list = []
    source = str(source)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt, jit, onnx, engine = (model.stride, model.names, model.pt, model.jit, model.onnx, model.engine)
    imgsz = check_img_size(imgsz, s=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz), half=False)
    dt, seen = ([0.0, 0.0, 0.0], 0)
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()
        dt[0] += t2 - t1
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(prediction=pred, classes=classes)
        dt[2] += time_sync() - t3
        for i, det in enumerate(pred):
            seen += 1
            p, im0, frame = (path, im0s.copy(), getattr(dataset, 'frame', 0))
            p = Path(p)
            s += '%gx%g ' % im.shape[2:]
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f'{n} {names[int(c)]}{'s' * (n > 1)}, '
                for *xyxy, conf, cls in reversed(det):
                    xyxy_ = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    xyxy_ = [int(x) for x in xyxy_]
                    top_left, bottom_right = ((xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3]))
                    bbox = [top_left, bottom_right]
                    c = int(cls)
                    label = names[c]
                    bbox_list.append(Bbox(bbox, label))
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    t = tuple((x / seen * 1000.0 for x in dt))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return bbox_list

def check_img_size(imgsz, s=32, floor=0):
    if isinstance(imgsz, int):
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    min_wh, max_wh = (2, 7680)
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = (x[:, :4] + c, x[:, 4])
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and 1 < n < 3000.0:
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output[xi] = x[i]
        if time.time() - t > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

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

def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(274, 1)
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE, 6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[274]
            image.info['exif'] = exif.tobytes()
    return image

