# Cluster 8

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

def get_mask_s(shape=None):
    canvas = np.zeros((512, 512), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 15
    thickness = 50
    text_size = cv2.getTextSize('S', font, font_scale, thickness=thickness)[0]
    text_x = (canvas.shape[1] - text_size[0]) // 2 + thickness // 2
    text_y = (canvas.shape[0] + text_size[1]) // 2 - thickness // 5
    canvas = cv2.putText(canvas, 'S', (text_x, text_y), font, font_scale, 255, thickness, cv2.LINE_AA)
    if shape:
        canvas = cv2.resize(canvas, shape[::-1])
    mask_s = torch.from_numpy(canvas > 128)
    return mask_s

