# Cluster 24

class LoadImages:

    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = (len(images), len(videos))
        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            self.count += 1
            img0 = cv2.imread(path)
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return (path, img, img0, self.cap, s)

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])
    if auto:
        dw, dh = (np.mod(dw, stride), np.mod(dh, stride))
    elif scaleFill:
        dw, dh = (0.0, 0.0)
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = (int(round(dh - 0.1)), int(round(dh + 0.1)))
    left, right = (int(round(dw - 0.1)), int(round(dw + 0.1)))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return (im, ratio, (dw, dh))

class LoadWebcam:

    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return (img_path, img, img0, None, s)

    def __len__(self):
        return 0

class LoadStreams:

    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = ([None] * n, [0] * n, [0] * n, [None] * n)
        self.sources = [clean_str(x) for x in sources]
        self.auto = auto
        for i, s in enumerate(sources):
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype='mp4').url
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=[i, cap, s], daemon=True)
            LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        n, f, read = (0, self.frames[i], 1)
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)
            time.sleep(1 / self.fps[i])

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all((x.is_alive() for x in self.threads)) or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        return (self.sources, img, img0, None, '')

    def __len__(self):
        return len(self.sources)

def clean_str(s):
    return re.sub(pattern='[|@#!¡·$€%&()=?¿^*;:,¨´><+]', repl='_', string=s)

