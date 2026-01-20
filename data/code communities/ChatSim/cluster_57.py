# Cluster 57

def setup_args(parser):
    parser.add_argument('--input_img', type=str, required=True, help='Path to a single input img')
    parser.add_argument('--point_coords', type=float, nargs='+', required=True, help='The coordinate of the point prompt, [coord_W coord_H].')
    parser.add_argument('--point_labels', type=int, nargs='+', required=True, help='The labels of the point prompt, 1 or 0.')
    parser.add_argument('--dilate_kernel_size', type=int, default=None, help='Dilate kernel size. Default: None')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path to the directory with results.')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help="The type of sam model to load. Default: 'vit_h")
    parser.add_argument('--sam_ckpt', type=str, required=True, help='The path to the SAM checkpoint to use for mask generation.')

def predict_masks_with_sam(img: np.ndarray, point_coords: List[List[float]], point_labels: List[int], model_type: str, ckpt_p: str, device='cuda'):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return (masks, scores, logits)

def fill_img_with_sd(img: np.ndarray, mask: np.ndarray, text_prompt: str, device='cuda'):
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float32).to(device)
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    img_crop_filled = pipe(prompt=text_prompt, image=Image.fromarray(img_crop), mask_image=Image.fromarray(mask_crop)).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled

@torch.no_grad()
def inpaint_img_with_lama(img: np.ndarray, mask: np.ndarray, config_p: str, ckpt_p: str, mod=8, device='cuda'):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.0)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    batch = model(batch)
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def replace_img_with_sd(img: np.ndarray, mask: np.ndarray, text_prompt: str, step: int=50, device='cuda'):
    pipe = StableDiffusionInpaintPipeline.from_pretrained('stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float32).to(device)
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)
    img_padded = pipe(prompt=text_prompt, image=Image.fromarray(img_padded), mask_image=Image.fromarray(255 - mask_padded), num_inference_steps=step).images[0]
    height, width, _ = img.shape
    img_resized, mask_resized = recover_size(np.array(img_padded), mask_padded, (height, width), padding_factors)
    mask_resized = np.expand_dims(mask_resized, -1) / 255
    img_resized = img_resized * (1 - mask_resized) + img * mask_resized
    return img_resized

@torch.no_grad()
def inpaint_video_with_builded_sttn(model, frames: List[Image.Image], masks: List[Image.Image], device='cuda') -> List[Image.Image]:
    w, h = (432, 240)
    neighbor_stride = 5
    video_length = len(frames)
    feats = [frame.resize((w, h)) for frame in frames]
    feats = _to_tensors(feats).unsqueeze(0) * 2 - 1
    _masks = [mask.resize((w, h), Image.NEAREST) for mask in masks]
    _masks = _to_tensors(_masks).unsqueeze(0)
    feats, _masks = (feats.to(device), _masks.to(device))
    comp_frames = [None] * video_length
    feats = (feats * (1 - _masks).float()).view(video_length, 3, h, w)
    feats = model.encoder(feats)
    _, c, feat_h, feat_w = feats.size()
    feats = feats.view(1, video_length, c, feat_h, feat_w)
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
        ref_ids = get_ref_index(neighbor_ids, video_length)
        pred_feat = model.infer(feats[0, neighbor_ids + ref_ids, :, :, :], _masks[0, neighbor_ids + ref_ids, :, :, :])
        pred_img = model.decoder(pred_feat[:len(neighbor_ids), :, :, :])
        pred_img = torch.tanh(pred_img)
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img.permute(0, 2, 3, 1) * 255
        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            b_mask = _masks.squeeze()[idx].unsqueeze(-1)
            b_mask = (b_mask != 0).int()
            frame = torch.from_numpy(np.array(frames[idx].resize((w, h))))
            frame = frame.to(device)
            img = pred_img[i] * b_mask + frame * (1 - b_mask)
            img = img.cpu().numpy()
            if comp_frames[idx] is None:
                comp_frames[idx] = img
            else:
                comp_frames[idx] = comp_frames[idx] * 0.5 + img * 0.5
    ori_w, ori_h = frames[0].size
    for idx in range(len(frames)):
        frame = np.array(frames[idx])
        b_mask = np.uint8(np.array(masks[idx])[..., np.newaxis] != 0)
        comp_frame = np.uint8(comp_frames[idx])
        comp_frame = Image.fromarray(comp_frame).resize((ori_w, ori_h))
        comp_frame = np.array(comp_frame)
        comp_frame = comp_frame * b_mask + frame * (1 - b_mask)
        comp_frames[idx] = Image.fromarray(np.uint8(comp_frame))
    return comp_frames

def get_ref_index(neighbor_ids, length):
    ref_length = 10
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index

@torch.no_grad()
def inpaint_video_with_sttn(video_p, mask_dir, output_dir, ckpt_p, model_type='sttn'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_sttn_model(ckpt_p, model_type, device)
    frames = read_frame_from_videos(video_p)
    masks = read_mask(mask_dir)
    comp_frames = inpaint_video_with_builded_sttn(model, frames, masks, device)
    video_stem = Path(video_p).stem
    output_p = Path(output_dir) / video_stem / f'removed_w_mask.mp4'
    output_p.parent.mkdir(exist_ok=True, parents=True)
    w, h = frames[0].size
    fps = imageio.v3.immeta(video_p, exclude_applied=False)['fps']
    writer = cv2.VideoWriter(str(output_p), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for idx in range(len(comp_frames)):
        writer.write(cv2.cvtColor(np.uint8(comp_frames[idx]), cv2.COLOR_BGR2RGB))
    writer.release()
    print(output_p)

def build_sttn_model(ckpt_p, model_type='sttn', device='cuda'):
    net = importlib.import_module(f'model.{model_type}')
    model = net.InpaintGenerator().to(device)
    data = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(data['netG'])
    model.eval()
    return model

def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames

def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames:
        m = Image.open(os.path.join(mpath, m))
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

class RemoveAnythingVideo(nn.Module):

    def __init__(self, args, tracker_target='ostrack', segmentor_target='sam', inpainter_target='sttn'):
        super().__init__()
        tracker_build_args = {'tracker_param': args.tracker_ckpt}
        segmentor_build_args = {'model_type': args.sam_model_type, 'ckpt_p': args.sam_ckpt}
        inpainter_build_args = {'lama': {'lama_config': args.lama_config, 'lama_ckpt': args.lama_ckpt}, 'sttn': {'model_type': 'sttn', 'ckpt_p': args.vi_ckpt}}
        self.tracker = self.build_tracker(tracker_target, **tracker_build_args)
        self.segmentor = self.build_segmentor(segmentor_target, **segmentor_build_args)
        self.inpainter = self.build_inpainter(inpainter_target, **inpainter_build_args[inpainter_target])
        self.tracker_target = tracker_target
        self.segmentor_target = segmentor_target
        self.inpainter_target = inpainter_target

    def build_tracker(self, target, **kwargs):
        assert target == 'ostrack', 'Only support sam now.'
        return build_ostrack_model(**kwargs)

    def build_segmentor(self, target='sam', **kwargs):
        assert target == 'sam', 'Only support sam now.'
        return build_sam_model(**kwargs)

    def build_inpainter(self, target='sttn', **kwargs):
        if target == 'lama':
            return build_lama_model(**kwargs)
        elif target == 'sttn':
            return build_sttn_model(**kwargs)
        else:
            raise NotImplementedError('Only support lama and sttn')

    def forward_tracker(self, frames_ps, init_box):
        init_box = np.array(init_box).astype(np.float32).reshape(-1, 4)
        seq = Sequence('tmp', frames_ps, 'inpaint-anything', init_box)
        all_box_xywh = get_box_using_ostrack(self.tracker, seq)
        return all_box_xywh

    def forward_segmentor(self, img, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        self.segmentor.set_image(img)
        masks, scores, logits = self.segmentor.predict(point_coords=point_coords, point_labels=point_labels, box=box, mask_input=mask_input, multimask_output=multimask_output, return_logits=return_logits)
        self.segmentor.reset_image()
        return (masks, scores)

    def forward_inpainter(self, frames, masks):
        print(self.inpainter_target)
        if self.inpainter_target == 'lama':
            for idx in range(len(frames)):
                frames[idx] = inpaint_img_with_builded_lama(self.inpainter, frames[idx], masks[idx], device=self.device)
        elif self.inpainter_target == 'sttn':
            frames = [Image.fromarray(frame) for frame in frames]
            masks = [Image.fromarray(np.uint8(mask * 255)) for mask in masks]
            frames = inpaint_video_with_builded_sttn(self.inpainter, frames, masks, device=self.device)
        else:
            raise NotImplementedError
        return frames

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def mask_selection(self, masks, scores, ref_mask=None, interactive=False):
        if interactive:
            raise NotImplementedError
        else:
            if ref_mask is not None:
                mse = np.mean((masks.astype(np.int32) - ref_mask.astype(np.int32)) ** 2, axis=(-2, -1))
                idx = mse.argmin()
            else:
                idx = scores.argmax()
            return masks[idx]

    @staticmethod
    def get_box_from_mask(mask):
        x, y, w, h = cv2.boundingRect(mask)
        return np.array([x, y, w, h])

    def forward(self, frame_ps: List[str], key_frame_idx: int, key_frame_point_coords: np.ndarray, key_frame_point_labels: np.ndarray, key_frame_mask_idx: int=None, dilate_kernel_size: int=15):
        """
        Mask is 0-1 ndarray in default
        Frame is 0-255 ndarray in default
        """
        assert key_frame_idx == 0, 'Only support key frame at the beginning.'
        key_frame_p = frame_ps[key_frame_idx]
        key_frame = iio.imread(key_frame_p)
        key_masks, key_scores = self.forward_segmentor(key_frame, key_frame_point_coords, key_frame_point_labels)
        if key_frame_mask_idx is not None:
            key_mask = key_masks[key_frame_mask_idx]
        else:
            key_mask = self.mask_selection(key_masks, key_scores)
        if dilate_kernel_size is not None:
            key_mask = dilate_mask(key_mask, dilate_kernel_size)
        key_box = self.get_box_from_mask(key_mask)
        print('Tracking ...')
        all_box = self.forward_tracker(frame_ps, key_box)
        print('Segmenting ...')
        all_mask = [key_mask]
        all_frame = [key_frame]
        ref_mask = key_mask
        for frame_p, box in zip(frame_ps[1:], all_box[1:]):
            frame = iio.imread(frame_p)
            x, y, w, h = box
            sam_box = np.array([x, y, x + w, y + h])
            masks, scores = self.forward_segmentor(frame, box=sam_box)
            mask = self.mask_selection(masks, scores, ref_mask)
            if dilate_kernel_size is not None:
                mask = dilate_mask(mask, dilate_kernel_size)
            ref_mask = mask
            all_mask.append(mask)
            all_frame.append(frame)
        print('Inpainting ...')
        all_frame = self.forward_inpainter(all_frame, all_mask)
        return (all_frame, all_mask, all_box)

def build_lama_model(config_p: str, ckpt_p: str, device='cuda'):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model

@torch.no_grad()
def inpaint_img_with_builded_lama(model, img: np.ndarray, mask: np.ndarray, config_p=None, mod=8, device='cuda'):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.0)
    mask = torch.from_numpy(mask).float()
    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    batch = model(batch)
    cur_res = batch['inpainted'][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def main_worker():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckpt))
    model.eval()
    frames = read_frame_from_videos(args.video)
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]
    masks = read_mask(args.mask)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = (feats.to(device), masks.to(device))
    comp_frames = [None] * video_length
    with torch.no_grad():
        feats = model.encoder((feats * (1 - masks).float()).view(video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    print('loading videos and masks from: {}'.format(args.video))
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            pred_feat = model.infer(feats[0, neighbor_ids + ref_ids, :, :, :], masks[0, neighbor_ids + ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[idx] + frames[idx] * (1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
    writer = cv2.VideoWriter(f'{args.mask}_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), default_fps, (w, h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(np.uint8) * binary_masks[f] + frames[f] * (1 - binary_masks[f])
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f'{args.mask}_result.mp4'))

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])
        with open(os.path.join(args['data_root'], args['name'], split + '.json'), 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug or split != 'train':
            self.video_names = self.video_names[:100]
        self._to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [f'{str(i).zfill(5)}.jpg' for i in range(self.video_dict[video_name])]
        all_masks = create_random_shape_with_random_motion(len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        frames = []
        masks = []
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return (frame_tensors, mask_tensors)

def get_inpainted_img(img, mask0, mask1, mask2):
    lama_config = args.lama_config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = []
    for mask in [mask0, mask1, mask2]:
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        img_inpainted = inpaint_img_with_builded_lama(model['lama'], img, mask, lama_config, device=device)
        out.append(img_inpainted)
    return out

def video2frames(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    initial_img = None
    for idx in tqdm(range(frame_num), 'Extract frames'):
        success, image = video.read()
        if idx == 0:
            initial_img = image.copy()
        assert success, 'extract the {}th frame in video {} failed!'.format(idx, video_path)
        cv2.imwrite('{}/{:05d}.jpg'.format(frame_path, idx), image)
    return (fps, initial_img)

def frames2video(frames_list, video_path, fps=30, remove_tmp=False):
    if isinstance(frames_list, str):
        frames_list = glob(f'{frames_list}/*.jpg')
    video_dir = os.path.dirname(video_path)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    writer = imageio.get_writer(video_dir, fps=fps, plugin='ffmpeg')
    for frame in tqdm(frames_list, 'Export video'):
        if isinstance(frame, str):
            frame = imageio.imread(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imageio.core.util.Array(frame)
        writer.append_data(frame)
    writer.close()
    print(f'find video at {video_path}.')
    if remove_tmp and isinstance(frames_list, str):
        shutil.rmtree(frames_list)

def video2seq(video_path, point_coords, point_labels, sam_model_type, sam_ckpt, output_dir):
    video_name, _ = os.path.splitext(video_path.split('/')[-1])
    frames_path = f'{output_dir}/{video_name}/original_frames'
    fps, first_frame = video2frames(video_path, frames_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    masks, scores, _ = predict_masks_with_sam(first_frame, [point_coords], point_labels, model_type=sam_model_type, ckpt_p=sam_ckpt, device=device)
    mask = masks[np.argmax(scores)][:, :, None].astype(np.uint8) * 255
    mask_loc = np.where(mask > 0)
    x1, x2 = (np.min(mask_loc[1]), np.max(mask_loc[1]))
    y1, y2 = (np.min(mask_loc[0]), np.max(mask_loc[0]))
    x1, y1, x2, y2 = list(map(lambda x: int(x), [x1, y1, x2, y2]))
    gt_rect = np.array([x1, y1, x2 - x1, y2 - y1]).astype(np.float32).reshape(-1, 4)
    frames_list = [os.path.join(frames_path, frame) for frame in os.listdir(frames_path) if frame.endswith('.jpg')]
    return (Sequence(video_name, frames_list, 'inpaint-anything', gt_rect.reshape(-1, 4)), fps)

