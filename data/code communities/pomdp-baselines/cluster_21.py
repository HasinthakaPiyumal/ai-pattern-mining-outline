# Cluster 21

class ImageEncoder(nn.Module):

    def __init__(self, image_shape, embed_size=100, depths=[8, 16], kernel_size=2, stride=1, activation=relu_name, from_flattened=False, normalize_pixel=False):
        super(ImageEncoder, self).__init__()
        self.shape = image_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.depths = [image_shape[0]] + depths
        layers = []
        h_w = self.shape[-2:]
        for i in range(len(self.depths) - 1):
            layers.append(nn.Conv2d(self.depths[i], self.depths[i + 1], kernel_size, stride))
            layers.append(ACTIVATIONS[activation]())
            h_w = conv_output_shape(h_w, kernel_size, stride)
        self.cnn = nn.Sequential(*layers)
        self.linear = nn.Linear(h_w[0] * h_w[1] * self.depths[-1], embed_size)
        self.from_flattened = from_flattened
        self.normalize_pixel = normalize_pixel
        self.embed_size = embed_size

    def forward(self, image):
        if self.from_flattened:
            batch_size = image.shape[:-1]
            img_shape = [np.prod(batch_size)] + list(self.shape)
            image = torch.reshape(image, img_shape)
        else:
            batch_size = [image.shape[0]]
        if self.normalize_pixel:
            image = image / 255.0
        embed = self.cnn(image)
        embed = torch.reshape(embed, list(batch_size) + [-1])
        embed = self.linear(embed)
        return embed

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor((h_w[0] + 2 * pad - dilation * (kernel_size[0] - 1) - 1) / stride + 1)
    w = floor((h_w[1] + 2 * pad - dilation * (kernel_size[1] - 1) - 1) / stride + 1)
    return (h, w)

