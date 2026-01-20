# Cluster 15

class TFConv(keras.layers.Layer):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        super().__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        assert isinstance(k, int), 'Convolution with multiple kernels are not allowed.'
        conv = keras.layers.Conv2D(c2, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False if hasattr(w, 'bn') else True, kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()), bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        if isinstance(w.act, nn.LeakyReLU):
            self.act = (lambda x: keras.activations.relu(x, alpha=0.1)) if act else tf.identity
        elif isinstance(w.act, nn.Hardswish):
            self.act = (lambda x: x * tf.nn.relu6(x + 3) * 0.166666667) if act else tf.identity
        elif isinstance(w.act, (nn.SiLU, SiLU)):
            self.act = (lambda x: keras.activations.swish(x)) if act else tf.identity
        else:
            raise Exception(f'no matching TensorFlow activation found for {w.act}')

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))

