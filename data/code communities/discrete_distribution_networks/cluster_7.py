# Cluster 7

class DDNInference:

    def __init__(self, weight_path, hw=(256, 256)):
        self.net = load_net(weight_path)
        self.hw = hw

    def inference(self, inf_arg):
        pass

    def process_np_img(self, img):
        if img.shape[0] != self.hw[0] or img.shape[1] != self.hw[1]:
            img = crop_and_resize(img, self.hw)
        if img.ndim == 2:
            img = np.concatenate([img[..., None]] * 3, -1)
        return uint8_to_tensor(img)

    def coloring_demo_inference(self, condition_rgb, n_samples=1, guided_rgba=None, clip_prompt=None):
        condition_source = self.process_np_img(condition_rgb)
        samplers = []
        if guided_rgba is not None and guided_rgba[..., -1].sum() > 1:
            guided_rgba = self.process_np_img(guided_rgba)
            if 'align color brightness' and 0:
                print('align color brightness')
                guided_normalized_tensor = (guided_rgba[:3] + 1) / 2
                brightness_guided = guided_normalized_tensor.mean(0, keepdim=True)
                brightness_condition = ((condition_source + 1) / 2).mean(0, keepdim=True)
                new_guided_normalized_tensor = guided_normalized_tensor * brightness_condition / (brightness_guided + 1e-05)
                guided_rgba[:3] = new_guided_normalized_tensor.clip(0, 1) * 2 - 1
            rgba_sampler = DistanceSamplerWithAlphaChannelTopk(guided_rgba)
            samplers.append(rgba_sampler)
        if clip_prompt is not None and clip_prompt not in ['', 'null']:
            clip_sampler = CLIPSampler(clip_prompt, keep_model_in_memory=True)
            samplers.append(clip_sampler)
        d_init = dict(condition_source=torch.cat([condition_source[None]] * n_samples))
        if len(samplers) == 1:
            batch_sampler = BatchedGuidedSampler(samplers[0])
            d_init['sampler'] = batch_sampler
        elif len(samplers) > 1:
            batch_sampler = MultiGuidedSampler({s: 1 / len(samplers) for s in samplers})
            d_init['sampler'] = batch_sampler
        d = self.net(d_init)
        stage_last_predicts = {'%sx%s' % pred.shape[-2:]: pred for pred in d['predicts']}
        stage_last_predicts_np = {k: list(t2rgb(v)) for k, v in stage_last_predicts.items()}
        d['stage_last_predicts_np'] = stage_last_predicts_np
        boxx.mg()
        return d

def load_net(path, device='cuda'):
    path = path.replace('https://oss.iap.hh-d.brainpp.cn', 's3:/').replace('http://localhost:58000/ddm_exps/', 'exps/')
    with open(path, 'rb') as f:
        if path.endswith('.pkl'):
            net = pickle.load(f)['ema'].to(device)
        elif path.endswith('.pt'):
            net = torch.load(f)['net'].to(device)
        net = net.eval()
    return net

def crop_and_resize(img: np.ndarray, out_hw=(256, 256)) -> np.ndarray:
    """
    Center-crop an image to match the aspect ratio of `out_hw`
    and then resize it. Works for any rectangle, not just squares.
    """
    target_h, target_w = out_hw
    target_ratio = target_w / target_h
    h, w = img.shape[:2]
    in_ratio = w / h
    if (h, w) == out_hw:
        return img
    if np.isclose(in_ratio, target_ratio, rtol=0, atol=1e-06):
        crop = img
    elif in_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        crop = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        crop = img[y0:y0 + new_h, :]
    pil_img = PIL.Image.fromarray(crop)
    resized = pil_img.resize(out_hw, PIL.Image.LANCZOS)
    return np.asarray(resized, dtype=img.dtype)

def uint8_to_tensor(img):
    target = img / 255 * 2 - 1
    target = target.transpose(2, 0, 1)
    target = torch.from_numpy(target).cuda().float()
    return target

class ReconstructionDatasetSampler:

    def __init__(self, dataset=None):
        """
        dataset[k] will return one h,w,3 RGB numpy image
        """
        if dataset is None:

            class CIFAR10Uint8(torchvision.datasets.cifar.CIFAR10):

                def __getitem__(self, idx):
                    pil, label = super().__getitem__(idx % 10000)
                    return np.array(pil)
            dataset = CIFAR10Uint8(os.path.expanduser('~/dataset'), train=False)
        self.dataset = dataset

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        if 'idx_gen' in dic:
            data_idx = int(dic['idx_gen'])
        else:
            data_idx = random.randint(0, len(self.dataset) - 1)
        if 'sampler_context' not in dic:
            img = self.dataset[data_idx % len(self.dataset)]
            if isinstance(img, tuple):
                img = img[0]
            if img.shape[-1] > 3 and img.shape[-3] <= 3:
                img = img.transpose(1, 2, 0)
            context = dict(target=uint8_to_tensor(img)[None], data_idx=data_idx)
        else:
            context = dic['sampler_context']
        resized = nn.functional.interpolate(context['target'], (h, w), mode='area')
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=int(probs.argmax()), condition0=context['target'], sampler_context=context)

