# Cluster 16

def random_perspective(im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)):
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    M = T @ S @ R @ P @ C
    if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    n = len(targets)
    if n:
        use_segments = any((x.any() for x in segments))
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                new[i] = segment2box(xy, width, height)
        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.1)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return (im, targets)

def resample_segments(segments, n=1000):
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T
    return segments

def segment2box(segment, width=640, height=640):
    x, y = segment.T
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y = (x[inside], y[inside])
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))

def copy_paste(im, labels, segments, p=0.5):
    n = len(segments)
    if p and n:
        h, w, c = im.shape
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = (labels[j], segments[j])
            box = (w - l[3], l[2], w - l[1], l[4])
            ioa = bbox_ioa(box, labels[:, 1:5])
            if (ioa < 0.3).all():
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)
        i = result > 0
        im[i] = result[i]
    return (im, labels, segments)

def bbox_ioa(box1, box2, eps=1e-07):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """
    box2 = box2.transpose()
    b1_x1, b1_y1, b1_x2, b1_y2 = (box1[0], box1[1], box1[2], box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = (box2[0], box2[1], box2[2], box2[3])
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter_area / box2_area

def cutout(im, labels, p=0.5):
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])
                labels = labels[ioa < 0.6]
    return labels

class LoadImagesAndLabels(Dataset):
    cache_version = 0.6

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and (not self.rect)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted((x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS))
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
        self.label_files = img2label_paths(self.img_files)
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = (np.load(cache_path, allow_pickle=True).item(), True)
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = (self.cache_labels(cache_path, prefix), False)
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())
        n = len(shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = range(n)
        include_class = []
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0
        if self.rect:
            s = self.shapes
            ar = s[:, 1] / s[:, 0]
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = (ari.min(), ari.max())
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        self.imgs, self.img_npy = ([None] * n, [None] * n)
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0
            self.img_hw0, self.img_hw = ([None] * n, [None] * n)
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1000000000.0:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        x = {}
        nm, nf, ne, nc, msgs = (0, 0, 0, 0, [])
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))), desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted'
        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = (nf, nm, ne, nc, len(self.img_files))
        x['msgs'] = msgs
        x['version'] = self.cache_version
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            img, labels = load_mosaic(self, index)
            shapes = None
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))
        else:
            img, (h0, w0), (h, w) = load_image(self, index)
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = ((h0, w0), ((h / h0, w / w0), pad))
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            if self.augment:
                img, labels = random_perspective(img, labels, degrees=hyp['degrees'], translate=hyp['translate'], scale=hyp['scale'], shear=hyp['shear'], perspective=hyp['perspective'])
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=0.001)
        if self.augment:
            img, labels = self.albumentations(img, labels)
            nl = len(labels)
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return (torch.from_numpy(img), labels_out, self.img_files[index], shapes)

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return (torch.stack(img, 0), torch.cat(label, 0), path, shapes)

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = ([], [], path[:n], shapes[:n])
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])
        for i in range(n):
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)
        for i, l in enumerate(label4):
            l[:, 0] = i
        return (torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4)

def mixup(im, labels, im2, labels2):
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return (im, labels)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if clip:
        clip_coords(x, (h - eps, w - eps))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2 / w
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2 / h
    y[:, 2] = (x[:, 2] - x[:, 0]) / w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h
    return y

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = (x * r[0] % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)

def load_mosaic(self, index):
    labels4, segments4 = ([], [])
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)
        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = (max(xc - w, 0), max(yc - h, 0), xc, yc)
            x1b, y1b, x2b, y2b = (w - (x2a - x1a), h - (y2a - y1a), w, h)
        elif i == 1:
            x1a, y1a, x2a, y2a = (xc, max(yc - h, 0), min(xc + w, s * 2), yc)
            x1b, y1b, x2b, y2b = (0, h - (y2a - y1a), min(w, x2a - x1a), h)
        elif i == 2:
            x1a, y1a, x2a, y2a = (max(xc - w, 0), yc, xc, min(s * 2, yc + h))
            x1b, y1b, x2b, y2b = (w - (x2a - x1a), 0, w, min(y2a - y1a, h))
        elif i == 3:
            x1a, y1a, x2a, y2a = (xc, yc, min(xc + w, s * 2), min(s * 2, yc + h))
            x1b, y1b, x2b, y2b = (0, 0, min(w, x2a - x1a), min(y2a - y1a, h))
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b
        labels, segments = (self.labels[index].copy(), self.segments[index].copy())
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4, degrees=self.hyp['degrees'], translate=self.hyp['translate'], scale=self.hyp['scale'], shear=self.hyp['shear'], perspective=self.hyp['perspective'], border=self.mosaic_border)
    return (img4, labels4)

def load_image(self, i):
    im = self.imgs[i]
    if im is None:
        npy = self.img_npy[i]
        if npy and npy.exists():
            im = np.load(npy)
        else:
            path = self.img_files[i]
            im = cv2.imread(path)
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 and (not self.augment) else cv2.INTER_LINEAR)
        return (im, (h0, w0), im.shape[:2])
    else:
        return (self.imgs[i], self.img_hw0[i], self.img_hw[i])

def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw
    y[:, 1] = h * x[:, 1] + padh
    return y

def load_mosaic9(self, index):
    labels9, segments9 = ([], [])
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)
    random.shuffle(indices)
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)
        if i == 0:
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
            h0, w0 = (h, w)
            c = (s, s, s + w, s + h)
        elif i == 1:
            c = (s, s - h, s + w, s)
        elif i == 2:
            c = (s + wp, s - h, s + wp + w, s)
        elif i == 3:
            c = (s + w0, s, s + w0 + w, s + h)
        elif i == 4:
            c = (s + w0, s + hp, s + w0 + w, s + hp + h)
        elif i == 5:
            c = (s + w0 - w, s + h0, s + w0, s + h0 + h)
        elif i == 6:
            c = (s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h)
        elif i == 7:
            c = (s - w, s + h0 - h, s, s + h0)
        elif i == 8:
            c = (s - w, s + h0 - hp - h, s, s + h0 - hp)
        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)
        labels, segments = (self.labels[index].copy(), self.segments[index].copy())
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]
        hp, wp = (h, w)
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])
    segments9 = [x - c for x in segments9]
    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)
    img9, labels9 = random_perspective(img9, labels9, segments9, degrees=self.hyp['degrees'], translate=self.hyp['translate'], scale=self.hyp['scale'], shear=self.hyp['shear'], perspective=self.hyp['perspective'], border=self.mosaic_border)
    return (img9, labels9)

