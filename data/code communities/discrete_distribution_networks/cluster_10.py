# Cluster 10

def tensor_to_imagenet_format(rgbs):
    img_tensor = (rgbs + 1) / 2
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(64, 64), mode='nearest')

    def normalize(tensor, mean, std):
        return (tensor - mean) / std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
    img_tensor_norm = normalize(img_tensor, mean, std)
    return img_tensor_norm

def normalize(tensor, mean, std):
    return (tensor - mean) / std

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

