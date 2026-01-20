# Cluster 22

def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
    b[:, 2:] = b[:, 2:] * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::1 if BGR else -1]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)
    return crop

def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and (not exist_ok):
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f'{path}{sep}*')
        matches = [re.search(f'%s{sep}(\\d+)' % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f'{path}{sep}{n}{suffix}')
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

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

