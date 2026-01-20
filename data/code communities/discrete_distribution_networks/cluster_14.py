# Cluster 14

def get_outputk(scalei, predict_c=3):
    k = 4 * get_channeln(scalei)
    k = min(max(16, k), 1024)
    if predict_c == 3:
        k = k // 2
    k = min(boxx.cf.get('kwargs', {}).get('max_outputk', k), k)
    return k

def get_channeln(scalei):
    channeln = 2 ** (13 - scalei)
    return min(max(4, channeln), 256)

@persistence.persistent_class
class PHDDNHandsDense(torch.nn.Module):

    def __init__(self, img_resolution=32, in_channels=3, out_channels=3, label_dim=0, augment_dim=0, model_channels=128, channel_mult=[1, 2, 2, 2], channel_mult_emb=4, num_blocks=4, attn_resolutions=[16], dropout=0.1, label_dropout=0, embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1, 1]):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-05)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-06, resample_filter=resample_filter, resample_proj=True, adaptive_scale=False, init=init, init_zero=init_zero, init_attn=init_attn)
        self.label_dim = label_dim
        condition_type = boxx.cf.get('kwargs', {}).get('condition')
        if condition_type:
            if condition_type == 'class':
                assert condition_type and label_dim, (condition_type, label_dim)
                condition_type += str(label_dim)
            self.condition_process = ConditionProcess(condition_type)
        self.condition_type = condition_type
        self.scalen = int(np.log2(img_resolution))
        self.module_names = []
        self.scale_to_module_names = {}
        self.scale_to_repeatn = {}
        if 'hands design':
            '\n            手工设计网络, 原则:\n                - 小 scale:\n                    - 背景: 小 scale 即低频信息, 低频信息一定是可以无损压缩的!,  所以需要大算力和表征空间的限制来让网络压缩低频信息. 但表征空间需要大于该尺度的实际信息量!\n                    - 减少 k, 增大 repeat 和 blockn * channeln, 因为 k 要多复用, 避免学不会和过拟合, 低维能有效分岔, 需要更多算力\n                - 大 scale\n                    - 高频信息难以无损压缩, 但可以通过更多的表示空间/采样 + 更多的采样来逼近高频信号, 以减缓平均模糊现象\n                    - 巨大的 k, 不要算力. 考更多 k 带来空间和随机性, 符合高频信号的随机性, 和低算力需求特性\n            '
            scale_to_channeln = [256, 256, 256, 256, 128, 64, 32]
            scale_to_blockn = [1, 8, 16, 16, 8, 4, 3]
            scale_to_repeatn = [3, 10, 10, 10, 10, 5, 2]
            scale_to_outputk = [64, 16, 16, 16, 64, 512, 512]

        def set_block(name, block):
            self.module_names.append(name)
            setattr(self, name, block)
            self.scale_to_module_names[scalei] = self.scale_to_module_names.get(scalei, []) + [name]
            return block
        start_size = boxx.cf.get('kwargs', {}).get('start_size', 1)
        blockn_times = boxx.cf.get('kwargs', {}).get('blockn_times', 1)
        self.scalis = range(int(math.log2(start_size)), self.scalen + 1)
        last_scalei = self.scalis[0]
        for scalei in self.scalis:
            size = 2 ** scalei
            channeln = get_channeln(scalei)
            last_channeln = get_channeln(scalei - 1)
            k = get_outputk(scalei)
            if last_scalei != scalei:
                block_up = UNetBlockWoEmb(in_channels=last_channeln, out_channels=channeln, up=True, **block_kwargs)
                set_block(f'block_{size}x{size}_0_up', DiscreteDistributionBlock(block_up, k, output_size=size))
            else:
                block = UNetBlockWoEmb(channeln, channeln, **block_kwargs)
                set_block(f'block_{size}x{size}_0', DiscreteDistributionBlock(block, k, output_size=size))
                if not scalei:
                    continue
            cin = channeln
            blockn = int(round(get_blockn(scalei) * blockn_times))
            for block_count in range(1, blockn):
                block = UNetBlockWoEmb(cin, channeln, **block_kwargs)
                set_block(f'block_{size}x{size}_{block_count}', DiscreteDistributionBlock(block, k, output_size=size))
                cin = channeln
        self.refiner_repeatn = 3 if boxx.cf.debug else boxx.cf.get('kwargs', {}).get('refinern', 0)
        refiner_outputk = 4
        if self.refiner_repeatn:
            unet = SongUNetInputDict(img_resolution=img_resolution, in_channels=channeln, out_channels=channeln, label_dim=label_dim, augment_dim=augment_dim, model_channels=model_channels, channel_mult=channel_mult, channel_mult_emb=channel_mult_emb, num_blocks=num_blocks, attn_resolutions=attn_resolutions, dropout=dropout, label_dropout=label_dropout, embedding_type=embedding_type, channel_mult_noise=channel_mult_noise, encoder_type=encoder_type, decoder_type=decoder_type, resample_filter=resample_filter)
            self.refiner = DiscreteDistributionBlock(unet, refiner_outputk, output_size=img_resolution, in_c=channeln, out_c=channeln, predict_c=out_channels, input_dict=True)

    def forward(self, d=None, _sigma=None, labels=None):
        if isinstance(d, torch.Tensor):
            d = {'target': d}
        elif d is None:
            d = {'batch_size': 1}
        assert isinstance(d, dict), d
        if self.label_dim and labels is not None:
            d['class_labels'] = labels
        for scalei in self.scalis:
            for repeati in range(self.scale_to_repeatn.get(scalei, 1)):
                for module_idx, name in enumerate(self.scale_to_module_names[scalei]):
                    if module_idx == 0 and repeati != 0:
                        continue
                    module = getattr(self, name)
                    d = module(d, condition_process=getattr(self, 'condition_process', None))
        feat = d['feat_last']
        batch_size = feat.shape[0]
        for repeati in range(self.refiner_repeatn):
            d['noise_labels'] = torch.Tensor([repeati / max(self.refiner_repeatn - 1, 1) * 2 - 1] * batch_size).to(feat)
            d = self.refiner(d, condition_process=getattr(self, 'condition_process', None))
        return d

    def table(self):
        times = 1
        mds = []
        for name in self.module_names:
            m = getattr(self, name)
            k = m.ddo.k if hasattr(m, 'ddo') else 1
            c = (m.in_c, m.out_c) if hasattr(m, 'in_c') else (None, None)
            size = m.output_size if hasattr(m, 'output_size') else size
            repeat = self.scale_to_repeatn.get(int(np.log2(size)), 1) if hasattr(m, 'output_size') else 1
            times *= k * repeat
            log2 = math.log2(times)
            row = dict(name=name, size=size, c=c, k=k, repeat=repeat, log2=log2, log10=math.log10(times))
            mds.append(row)
        return boxx.Markdown(mds)

    def get_sdds(self):
        sdds = []
        for name in self.module_names:
            m = getattr(self, name)
            sdds.append(m.ddo.sdd)
        return sdds

def get_blockn(scalei):
    scalei_to_blockn = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64}
    if scalei not in scalei_to_blockn:
        scalei_to_blockn[scalei] = max(scalei_to_blockn.values())
    blockn = scalei_to_blockn[scalei]
    blockn = min(boxx.cf.get('kwargs', {}).get('max_blockn', blockn), blockn)
    return blockn

@persistence.persistent_class
class PHDDNHandsSparse(PHDDNHandsDense):

    def __init__(self, img_resolution=32, in_channels=3, out_channels=3, label_dim=0, augment_dim=0, model_channels=128, channel_mult=[1, 2, 2, 2], channel_mult_emb=4, num_blocks=4, attn_resolutions=[16], dropout=0.1, label_dropout=0, embedding_type='positional', channel_mult_noise=1, encoder_type='standard', decoder_type='standard', resample_filter=[1, 1]):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        torch.nn.Module.__init__(self)
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-05)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-06, resample_filter=resample_filter, resample_proj=True, adaptive_scale=False, init=init, init_zero=init_zero, init_attn=init_attn)
        self.scalen = int(np.log2(img_resolution))
        self.module_names = []
        self.scale_to_module_names = {}
        self.scale_to_repeatn = {}
        if 'hands design':
            '\n            手工设计网络, 原则:\n                - 小 scale:\n                    - 背景: 小 scale 即低频信息, 低频信息一定是可以无损压缩的!,  所以需要大算力和表征空间的限制来让网络压缩低频信息. 但表征空间需要大于该尺度的实际信息量!\n                    - 减少 k, 增大 repeat 和 blockn * channeln, 因为 k 要多复用, 避免学不会和过拟合, 低维能有效分岔, 需要更多算力\n                - 大 scale\n                    - 高频信息难以无损压缩, 但可以通过更多的表示空间/采样 + 更多的采样来逼近高频信号, 以减缓平均模糊现象\n                    - 巨大的 k, 不要算力. 考更多 k 带来空间和随机性, 符合高频信号的随机性, 和低算力需求特性\n            '
            scale_to_channeln = [256, 256, 256, 256, 128, 64, 32]
            scale_to_blockn = [4, 8, 16, 16, 8, 5, 4]
            scale_to_repeatn = [2, 10, 10, 10, 10, 6, 3]
            scale_to_outputk = [64, 32, 32, 32, 64, 512, 512]
            if boxx.cf.debug:
                scale_to_channeln = [4, 8] * 7
            get_channeln = lambda scalei: scale_to_channeln[scalei]
            get_blockn = lambda scalei: scale_to_blockn[scalei]
            get_outputk = lambda scalei: scale_to_outputk[scalei]
            get_repeatn = lambda scalei: scale_to_repeatn[scalei]
            self.scale_to_repeatn = dict(enumerate(scale_to_repeatn))

        def set_block(name, block):
            self.module_names.append(name)
            setattr(self, name, block)
            self.scale_to_module_names[scalei] = self.scale_to_module_names.get(scalei, []) + [name]
            return block
        for scalei in range(self.scalen + 1):
            size = 2 ** scalei
            channeln = get_channeln(scalei)
            last_channeln = get_channeln(scalei - 1)
            k = get_outputk(scalei)
            if scalei:
                block_up = UpBlock(in_channels=last_channeln, out_channels=channeln, up=True, **block_kwargs)
                set_block(f'block_{size}x{size}_up', block_up)
            else:
                pass
            blocks = [UNetBlockWoEmb(channeln, channeln, **block_kwargs) for block_count in range(0, get_blockn(scalei))]
            blocks = torch.nn.Sequential(*blocks)
            dd_blocks = DiscreteDistributionBlock(blocks, k, output_size=size)
            set_block(f'blocks_{size}x{size}', dd_blocks)

    def forward(self, d=None, _sigma=None, labels=None):
        for scalei in range(self.scalen + 1):
            for repeati in range(self.scale_to_repeatn.get(scalei, 1)):
                for module_idx, name in enumerate(self.scale_to_module_names[scalei]):
                    if name.endswith('_up') and repeati != 0:
                        continue
                    module = getattr(self, name)
                    d = module(d)
        return d

