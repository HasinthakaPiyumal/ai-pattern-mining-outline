# Cluster 89

def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    from skimage import color
    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and (not mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.0
    return np2tensor(img_lab)

def np2tensor(np_obj):
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def tensorlab2tensor(lab_tensor, return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings('ignore')
    lab = tensor2np(lab_tensor) * 100.0
    lab[:, :, 0] = lab[:, :, 0] + 50
    rgb_back = 255.0 * np.clip(color.lab2rgb(lab.astype('float')), 0, 1)
    if return_inbnd:
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1.0 * np.isclose(lab_back, lab, atol=2.0)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return (im2tensor(rgb_back), mask)
    else:
        return im2tensor(rgb_back)

def tensor2np(tensor_obj):
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))

class DistModel(BaseModel):

    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None, use_gpu=True, printNet=False, spatial=False, is_train=False, lr=0.0001, beta1=0.5, version='0.1'):
        """
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
        """
        BaseModel.initialize(self, use_gpu=use_gpu)
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]' % (model, net)
        if self.model == 'net-lin':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net, use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = dict(map_location='cpu')
            if model_path is None:
                import inspect
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lpips_models', f'{net}.pth'))
            if not is_train:
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        elif self.model == 'net':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif self.model in ['L2', 'l2']:
            self.net = L2(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'L2'
        elif self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']:
            self.net = DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError('Model [%s] not recognized.' % self.model)
        self.trainable_parameters = list(self.net.parameters())
        if self.is_train:
            self.rankLoss = BCERankingLoss()
            self.trainable_parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.trainable_parameters, lr=lr, betas=(beta1, 0.999))
        else:
            self.net.eval()
        if printNet:
            print('---------- Networks initialized -------------')
            print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        """ Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        """
        return self.net(in0, in1, retPerLayer=retPerLayer)

    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if hasattr(module, 'weight') and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

    def forward_train(self):
        assert False, "We shoud've not get here when using LPIPS as a metric"
        self.d0 = self(self.var_ref, self.var_p0)
        self.d1 = self(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)
        self.var_judge = Variable(1.0 * self.input_judge).view(self.d0.size())
        self.loss_total = self.rankLoss(self.d0, self.d1, self.var_judge * 2.0 - 1.0)
        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        """ d0, d1 are Variables, judge is a Tensor """
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()), ('acc_r', self.acc_r)])
        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])
        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]
        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)
        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)
        return OrderedDict([('ref', ref_img_vis), ('p0', p0_img_vis), ('p1', p1_img_vis)])

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr
        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr

def tensor2im(image_tensor, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            return ret_var

def l2(p0, p1, range=255.0):
    return 0.5 * np.mean((p0 / range - p1 / range) ** 2)

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = dssim(1.0 * tensor2im(in0.data), 1.0 * tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = dssim(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        return ret_var

def dssim(p0, p1, range=255.0):
    return (1 - compare_ssim(p0, p1, data_range=range, multichannel=True)) / 2.0

def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    from skimage import color
    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and (not mc_only):
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.0
    return np2tensor(img_lab)

def tensorlab2tensor(lab_tensor, return_inbnd=False):
    from skimage import color
    import warnings
    warnings.filterwarnings('ignore')
    lab = tensor2np(lab_tensor) * 100.0
    lab[:, :, 0] = lab[:, :, 0] + 50
    rgb_back = 255.0 * np.clip(color.lab2rgb(lab.astype('float')), 0, 1)
    if return_inbnd:
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1.0 * np.isclose(lab_back, lab, atol=2.0)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return (im2tensor(rgb_back), mask)
    else:
        return im2tensor(rgb_back)

def im2tensor(image, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

class DistModel(BaseModel):

    def name(self):
        return self.model_name

    def initialize(self, model='net-lin', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None, use_gpu=True, printNet=False, spatial=False, is_train=False, lr=0.0001, beta1=0.5, version='0.1'):
        """
        INPUTS
            model - ['net-lin'] for linearly calibrated network
                    ['net'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
        """
        BaseModel.initialize(self, use_gpu=use_gpu)
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]' % (model, net)
        if self.model == 'net-lin':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net, use_dropout=True, spatial=spatial, version=version, lpips=True)
            kw = dict(map_location='cpu')
            if model_path is None:
                import inspect
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lpips_models', f'{net}.pth'))
            if not is_train:
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        elif self.model == 'net':
            self.net = PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif self.model in ['L2', 'l2']:
            self.net = L2(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'L2'
        elif self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']:
            self.net = DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError('Model [%s] not recognized.' % self.model)
        self.trainable_parameters = list(self.net.parameters())
        if self.is_train:
            self.rankLoss = BCERankingLoss()
            self.trainable_parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(self.trainable_parameters, lr=lr, betas=(beta1, 0.999))
        else:
            self.net.eval()
        if printNet:
            print('---------- Networks initialized -------------')
            print_network(self.net)
            print('-----------------------------------------------')

    def forward(self, in0, in1, retPerLayer=False):
        """ Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        """
        return self.net(in0, in1, retPerLayer=retPerLayer)

    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if hasattr(module, 'weight') and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data['ref']
        self.input_p0 = data['p0']
        self.input_p1 = data['p1']
        self.input_judge = data['judge']

    def forward_train(self):
        assert False, "We shoud've not get here when using LPIPS as a metric"
        self.d0 = self(self.var_ref, self.var_p0)
        self.d1 = self(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)
        self.var_judge = Variable(1.0 * self.input_judge).view(self.d0.size())
        self.loss_total = self.rankLoss(self.d0, self.d1, self.var_judge * 2.0 - 1.0)
        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        """ d0, d1 are Variables, judge is a Tensor """
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict([('loss_total', self.loss_total.data.cpu().numpy()), ('acc_r', self.acc_r)])
        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])
        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]
        ref_img = tensor2im(self.var_ref.data)
        p0_img = tensor2im(self.var_p0.data)
        p1_img = tensor2im(self.var_p1.data)
        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)
        return OrderedDict([('ref', ref_img_vis), ('p0', p0_img_vis), ('p1', p1_img_vis)])

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, '', label)
        else:
            self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr
        print('update lr [%s] decay: %f -> %f' % (type, self.old_lr, lr))
        self.old_lr = lr

class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = l2(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = dssim(1.0 * tensor2im(in0.data), 1.0 * tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = dssim(tensor2np(tensor2tensorlab(in0.data, to_norm=False)), tensor2np(tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        return ret_var

