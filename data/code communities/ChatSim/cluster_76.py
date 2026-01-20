# Cluster 76

class PerceptualLoss(nn.Module):

    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()
        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD
        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []
        for weights in vgg.parameters():
            weights.requires_grad = False
        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)
        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')
        losses = []
        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target
        for layer in self.vgg[:30]:
            features_input = layer(features_input)
            features_target = layer(features_target)
            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')
                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)
                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)
        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')
        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input
        features_input = self.vgg(features_input)
        return features_input

def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(f'{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}')

def visualize_mask_and_images_batch(batch: Dict[str, torch.Tensor], keys: List[str], max_items=10, last_without_mask=True, rescale_keys=None) -> np.ndarray:
    batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys or k == 'mask'}
    batch_size = next(iter(batch.values())).shape[0]
    items_to_vis = min(batch_size, max_items)
    result = []
    for i in range(items_to_vis):
        cur_dct = {k: tens[i] for k, tens in batch.items()}
        result.append(visualize_mask_and_images(cur_dct, keys, last_without_mask=last_without_mask, rescale_keys=rescale_keys))
    return np.concatenate(result, axis=0)

def visualize_mask_and_images(images_dict: Dict[str, np.ndarray], keys: List[str], last_without_mask=True, rescale_keys=None, mask_only_first=None, black_mask=False) -> np.ndarray:
    mask = images_dict['mask'] > 0.5
    result = []
    for i, k in enumerate(keys):
        img = images_dict[k]
        img = np.transpose(img, (1, 2, 0))
        if rescale_keys is not None and k in rescale_keys:
            img = img - img.min()
            img /= img.max() + 1e-05
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img_classes = img.argmax(2)
            img = color.label2rgb(img_classes, colors=COLORS)
        if mask_only_first:
            need_mark_boundaries = i == 0
        else:
            need_mark_boundaries = i < len(keys) - 1 or not last_without_mask
        if need_mark_boundaries:
            if black_mask:
                img = img * (1 - mask[0][..., None])
            img = mark_boundaries(img, mask[0], color=(1.0, 0.0, 0.0), outline_color=(1.0, 1.0, 1.0), mode='thick')
        result.append(img)
    return np.concatenate(result, axis=1)

class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10, last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        vis_img = visualize_mask_and_images_batch(batch, self.key_order, max_items=self.max_items_in_batch, last_without_mask=self.last_without_mask, rescale_keys=self.rescale_keys)
        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')
        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)

class PerceptualLoss(nn.Module):

    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()
        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD
        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []
        for weights in vgg.parameters():
            weights.requires_grad = False
        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)
        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')
        losses = []
        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target
        for layer in self.vgg[:30]:
            features_input = layer(features_input)
            features_target = layer(features_target)
            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')
                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)
                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)
        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')
        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input
        features_input = self.vgg(features_input)
        return features_input

def visualize_mask_and_images_batch(batch: Dict[str, torch.Tensor], keys: List[str], max_items=10, last_without_mask=True, rescale_keys=None) -> np.ndarray:
    batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items() if k in keys or k == 'mask'}
    batch_size = next(iter(batch.values())).shape[0]
    items_to_vis = min(batch_size, max_items)
    result = []
    for i in range(items_to_vis):
        cur_dct = {k: tens[i] for k, tens in batch.items()}
        result.append(visualize_mask_and_images(cur_dct, keys, last_without_mask=last_without_mask, rescale_keys=rescale_keys))
    return np.concatenate(result, axis=0)

class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10, last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        vis_img = visualize_mask_and_images_batch(batch, self.key_order, max_items=self.max_items_in_batch, last_without_mask=self.last_without_mask, rescale_keys=self.rescale_keys)
        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')
        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)

