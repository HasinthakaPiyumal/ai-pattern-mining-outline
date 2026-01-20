# Cluster 9

class L2MaskedSampler:

    def __init__(self, target, mask=0):
        self.raw = target
        h, w = target.shape[-2:]
        if isinstance(mask, int):
            mask_ = torch.zeros((h, w)) > 0
            if mask == 0:
                mask_[:, :w // 2] = 1
            if mask == 1:
                mask_[:h // 2] = 1
            if mask == 2:
                mask_[h // 2:] = 1
            if mask == 3:
                mask_[:h // 2, :w // 2] = 1
                mask_[h // 2:, w // 2:] = 1
            if mask == 4:
                mask_[h * 2 // 5:h * 5 // 9] = 1
            if mask == 5:
                mask_[h * 2 // 5:h * 5 // 9] = 1
                mask_[:, w * 2 // 5:w * 3 // 5] = 1
            if mask == 6:
                mask_[h // 4:h * 3 // 4, w // 4:w * 3 // 4] = 1
            if mask == 7:
                mask_ = get_mask_s((h, w))
            if mask == 8:
                mask_ = ~get_mask_s((h, w))
            if mask == 9:
                mask_[h * 2 // 5:h * 5 // 9] = 1
                mask_ = ~mask_
            mask = mask_
        self.mask = mask.cuda()
        self.target = target[None] * self.mask

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode='area')
        mask_resized = nn.functional.interpolate(self.mask[None, None].float(), (h, w), mode='area')
        loss = torch.abs(rgbs - resized)
        loss = (rgbs - resized) ** 2
        probs = nn.functional.softmax(-(loss * mask_resized).sum([-1, -2]).mean(-1) / (mask_resized.sum() + eps), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode='area')
        loss = (rgbs - self.target) ** 2
        mask_resized = self.mask[None, None].float()
        probs = nn.functional.softmax(-(loss * mask_resized).sum([-1, -2]).mean(-1) / (mask_resized.sum() + eps), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

def topk_sample(probs, topk):
    if isinstance(topk, float):
        topk = max(1, int(round(len(probs) * topk)))
    module = torch if isinstance(probs, torch.Tensor) else np
    args = module.argsort(probs)
    idx_k = int(random.choice(args[-topk:]))
    return idx_k

class NoiseSampler(L2Sampler):

    def __init__(self, target, noise_rate=0.53):
        self.raw = target
        self.target = target[None] + torch.randn_like(target)[None] * noise_rate

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode='area')
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class LowBitSampler(L2Sampler):

    def __init__(self, target, brightnessn=4):
        self.raw = target
        self.f = lambda x: (x * brightnessn / 2).round() * 2 / brightnessn
        self.target = self.f(target[None])

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode='area')
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

    def __call__2(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        resized = self.target
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode='nearest')
        probs = nn.functional.softmax(-((self.f(rgbs) - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class ColorfulSampler:

    def __init__(self, target):
        self.raw = target
        self.target = target[None].mean(-3, keepdims=True)

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        resized = nn.functional.interpolate(self.target, (h, w), mode='area')
        rgbs = rgbs.mean(-3, keepdims=True)
        probs = nn.functional.softmax(-((rgbs - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class SuperResSampler:

    def __init__(self, target, shape=0.5):
        self.raw = target
        self.target = boxx.resize(target[None], shape, 'area')

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        resized = nn.functional.interpolate(rgbs, (hh, ww), mode='area')
        probs = nn.functional.softmax(-((self.target - resized) ** 2).mean([-1, -2, -3]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class CannySampler:

    def __init__(self, target):
        self.raw = target
        self.f = lambda timg: cv2.Canny(t2rgb(timg), 100, 200)
        self.target = self.f(target)
        self.cache = {self.target.shape: self.target}

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, h, w, c = rgbs.shape
        hh, ww = self.target.shape[-2:]
        rgbs = nn.functional.interpolate(rgbs, (hh, ww), mode='nearest')
        rgbs = rgbs.permute(0, 2, 3, 1).cpu().numpy()
        target_resized = self.target
        resized_imgs = []
        for img in rgbs:
            resized_img = self.f(img)
            resized_imgs.append(resized_img)
        resized_imgs = np.stack(resized_imgs, axis=0)
        if not increase('edge show') % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        probs = resized_imgs[:, target_resized > 0].mean(-1) + (1 - resized_imgs[:, target_resized < 0.5].mean(-1))
        if h <= 16:
            probs += resized_imgs.mean(-1).mean(-1)
        g()
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class EdgeSampler:

    def __init__(self, target):
        self.raw = target
        self.sobel_kernel_horizontal = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[None, None].requires_grad_(False).cuda()
        self.sobel_kernel_vertical = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])[None, None].requires_grad_(False).cuda()
        self.target = self.sobel_operator(target[None])
        self.cache = {self.target.shape: self.target}

    def sobel_operator(self, img, h=None):
        grey = img.mean(-3, keepdim=True)
        if h is None:
            h, w = grey.shape[-2:]
        output_horizontal = F.conv2d(grey, self.sobel_kernel_horizontal)
        output_vertical = F.conv2d(grey, self.sobel_kernel_vertical)
        magnitude = torch.sqrt(output_horizontal ** 2 + output_vertical ** 2)
        thre = 1.5
        thre = 1.0
        edge = magnitude
        if h < 50:
            thre = 2
        if h < 20:
            thre = 2.1
        return edge

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        hh, ww = self.target.shape[-2:]
        target_resized = self.target
        if (h, w) not in self.cache and h > 600:
            show(target_resized, rgbs[0], t2rgb, self.sobel_operator(rgbs[:1]))
        rgbs = nn.functional.interpolate(rgbs, (hh + 2, ww + 2), mode='bilinear')
        resized_imgs = self.sobel_operator(rgbs, h=h)
        if 0:
            kernel = torch.ones((1, 1, 3, 3), device='cuda:0')
            convolved_imgs = F.conv2d(resized_imgs, kernel, padding=1)
            target_resized_expanded = target_resized.expand_as(convolved_imgs)
            masks = convolved_imgs == target_resized_expanded
        probs = -torch.abs(resized_imgs - target_resized).clip(0.0).float().mean(-1).mean(-1).mean(-1)
        idx_k = topk_sample(probs, 2)
        g()
        if not increase('edge show') % 8:
            show(resized_imgs[:1], rgbs[:1], t2rgb)
        return dict(probs=probs, idx_k=idx_k, condition0=self.target, condition_source0=resized_imgs[0])

class StyleTransferResnet:

    def __init__(self, target):
        self.model = StyleResNet.build_with_pretrain().eval().cuda()
        self.raw = target
        self.target = self.f(target[None])

    def f(self, rgbs):
        with torch.no_grad():
            return self.model(tensor_to_imagenet_format(rgbs))

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        feats = self.f(rgbs)
        probs = nn.functional.softmax(-((self.target - feats) ** 2).mean([-1]), 0)
        return dict(probs=probs, idx_k=topk_sample(probs, 2), condition0=self.target, condition_source0=self.raw)

class CifarSampler:

    def __init__(self, target=None, entropy=False):
        with impt('../../asset/resnet_cifar_zhk'):
            from cifar_pretrain import CifarPretrain
        self.model = CifarPretrain()
        self.target = target
        self.entropy = entropy

    def __call__(self, dic):
        rgbs = dic['rgbs']
        k, c, h, w = rgbs.shape
        rgbs = torch.nn.functional.interpolate(rgbs, size=(32, 32), mode='nearest')
        class_probs = nn.functional.softmax(self.model(rgbs), -1)
        target = self.target
        if target is None:
            if 'idx_gen' in dic:
                target = int(dic['idx_gen'] % 10)
        topk = 0.1
        if self.entropy:
            target = None
            topk = 0.1
        if target is None:
            probs = class_probs.max(1)[0]
        else:
            probs = class_probs[:, target]
        idx_k = topk_sample(probs, topk)
        mg()
        return dict(probs=probs, idx_k=idx_k, condition0=target)

class CLIPSampler:
    model_in_memory = None
    lock = threading.Lock()

    def __init__(self, target, keep_model_in_memory=False):
        import clip
        self.raw = target
        model_name = 'ViT-B/32'
        device = 'cuda'
        self.dtype = torch.float32
        with CLIPSampler.lock:
            if keep_model_in_memory and CLIPSampler.model_in_memory is not None:
                self.model = CLIPSampler.model_in_memory
            else:
                self.model, transform = clip.load(model_name, device=device)
                self.model = self.model.to(device)
                if keep_model_in_memory:
                    CLIPSampler.model_in_memory = self.model
        with torch.no_grad():
            tokens = clip.tokenize([target]).to(device)
            self.target = self.model.encode_text(tokens)

    def __call__(self, dic):
        rgbs = dic['rgbs']
        dtype = rgbs.dtype
        device = rgbs.device
        mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1).to(device)
        std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1).to(device)
        rgbs = ((rgbs + 1) / 2 - mean) / std
        rgbs = nn.functional.interpolate(rgbs, (224, 224), mode='bicubic')
        with torch.no_grad():
            num_minibatches = math.ceil(rgbs.size(0) / 8)
            minibatches = torch.chunk(rgbs, num_minibatches)
            image_features = []
            for batch in minibatches:
                image_features.append(self.model.encode_image(batch))
            image_features = torch.cat(image_features, dim=0)
            text_features = self.target
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100 * image_features @ text_features.T)[:, 0]
        mg()
        return dict(probs=probs, idx_k=topk_sample(probs, 1 if len(probs) == 2 else 2), condition_source0=self.raw, condition0=self.target)

class MultiGuidedSampler:

    def __init__(self, sampler_to_weight):
        self.sampler_to_weight = sampler_to_weight

    def __call__(self, d):
        s2w = self.sampler_to_weight
        sn = len(s2w)
        outputs = d['output']
        b, k, c, h, w = outputs.shape
        probsd = {}
        weighted_argsortd = {}
        for si, sampler in enumerate(s2w):
            dics = []
            for batchi in range(b):
                input_dic = dict(rgbs=outputs[batchi])
                if 'idx_gens' in d:
                    input_dic['idx_gen'] = d['idx_gens'][batchi]
                if 'sampler_contexts' in d:
                    input_dic['sampler_context'] = d['sampler_contexts'][si][batchi]
                dic = sampler(input_dic)
                dics.append(dic)
            probsd[sampler] = [npa(dic['probs']) for dic in dics]
            weighted_argsortd[sampler] = [np.argsort(npa(dic['probs'])).argsort() * s2w[sampler] for dic in dics]
            if 'condition_source0' in dic:
                d['condition_source0'] = d.get('condition_source0', {})
                d['condition_source0'][si] = [dic['condition_source0'] for dic in dics]
            if 'condition0' in dic:
                d['condition0'] = d.get('condition0', {})
                d['condition0'][si] = [dic['condition0'] for dic in dics]
            if 'sampler_context' in dic:
                d['sampler_contexts'] = d.get('sampler_contexts', {})
                d['sampler_contexts'][si] = [dic['sampler_context'] for dic in dics]
        weighted_argsort = npa(list(weighted_argsortd.values()))
        bk = weighted_argsort.sum(0)
        idx_ks = [topk_sample(prob, 2) for prob in bk]
        return npa(idx_ks)

