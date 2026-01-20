# Cluster 148

def log_txt_as_img(wh, xc, size=10):
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new('RGB', wh, color='white')
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = '\n'.join((xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc)))
        try:
            draw.text((0, 0), lines, fill='black', font=font)
        except UnicodeEncodeError:
            print('Cant encode string for logging. Skipping.')
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

class NoisyLatentImageClassifier(pl.LightningModule):

    def __init__(self, diffusion_path, num_classes, ckpt_path=None, pool='attention', label_key=None, diffusion_ckpt_path=None, scheduler_config=None, weight_decay=0.01, log_steps=10, monitor='val/loss', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        diffusion_config = natsorted(glob(os.path.join(diffusion_path, 'configs', '*-project.yaml')))[-1]
        self.diffusion_config = OmegaConf.load(diffusion_config).model
        self.diffusion_config.params.ckpt_path = diffusion_ckpt_path
        self.load_diffusion()
        self.monitor = monitor
        self.numd = self.diffusion_model.first_stage_model.encoder.num_resolutions - 1
        self.log_time_interval = self.diffusion_model.num_timesteps // log_steps
        self.log_steps = log_steps
        self.label_key = label_key if not hasattr(self.diffusion_model, 'cond_stage_key') else self.diffusion_model.cond_stage_key
        assert self.label_key is not None, 'label_key neither in diffusion model nor in model.params'
        if self.label_key not in __models__:
            raise NotImplementedError()
        self.load_classifier(ckpt_path, pool)
        self.scheduler_config = scheduler_config
        self.use_scheduler = self.scheduler_config is not None
        self.weight_decay = weight_decay

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location='cpu')
        if 'state_dict' in list(sd.keys()):
            sd = sd['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys')
        if len(missing) > 0:
            print(f'Missing Keys: {missing}')
        if len(unexpected) > 0:
            print(f'Unexpected Keys: {unexpected}')

    def load_diffusion(self):
        model = instantiate_from_config(self.diffusion_config)
        self.diffusion_model = model.eval()
        self.diffusion_model.train = disabled_train
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def load_classifier(self, ckpt_path, pool):
        model_config = deepcopy(self.diffusion_config.params.unet_config.params)
        model_config.in_channels = self.diffusion_config.params.unet_config.params.out_channels
        model_config.out_channels = self.num_classes
        if self.label_key == 'class_label':
            model_config.pool = pool
        self.model = __models__[self.label_key](**model_config)
        if ckpt_path is not None:
            print('#####################################################################')
            print(f'load from ckpt "{ckpt_path}"')
            print('#####################################################################')
            self.init_from_ckpt(ckpt_path)

    @torch.no_grad()
    def get_x_noisy(self, x, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x))
        continuous_sqrt_alpha_cumprod = None
        if self.diffusion_model.use_continuous_noise:
            continuous_sqrt_alpha_cumprod = self.diffusion_model.sample_continuous_noise_level(x.shape[0], t + 1)
        return self.diffusion_model.q_sample(x_start=x, t=t, noise=noise, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod)

    def forward(self, x_noisy, t, *args, **kwargs):
        return self.model(x_noisy, t)

    @torch.no_grad()
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_conditioning(self, batch, k=None):
        if k is None:
            k = self.label_key
        assert k is not None, 'Needs to provide label key'
        targets = batch[k].to(self.device)
        if self.label_key == 'segmentation':
            targets = rearrange(targets, 'b h w c -> b c h w')
            for down in range(self.numd):
                h, w = targets.shape[-2:]
                targets = F.interpolate(targets, size=(h // 2, w // 2), mode='nearest')
        return targets

    def compute_top_k(self, logits, labels, k, reduction='mean'):
        _, top_ks = torch.topk(logits, k, dim=1)
        if reduction == 'mean':
            return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
        elif reduction == 'none':
            return (top_ks == labels[:, None]).float().sum(dim=-1)

    def on_train_epoch_start(self):
        self.diffusion_model.model.to('cpu')

    @torch.no_grad()
    def write_logs(self, loss, logits, targets):
        log_prefix = 'train' if self.training else 'val'
        log = {}
        log[f'{log_prefix}/loss'] = loss.mean()
        log[f'{log_prefix}/acc@1'] = self.compute_top_k(logits, targets, k=1, reduction='mean')
        log[f'{log_prefix}/acc@5'] = self.compute_top_k(logits, targets, k=5, reduction='mean')
        self.log_dict(log, prog_bar=False, logger=True, on_step=self.training, on_epoch=True)
        self.log('loss', log[f'{log_prefix}/loss'], prog_bar=True, logger=False)
        self.log('global_step', self.global_step, logger=False, on_epoch=False, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, on_step=True, logger=True, on_epoch=False, prog_bar=True)

    def shared_step(self, batch, t=None):
        x, *_ = self.diffusion_model.get_input(batch, k=self.diffusion_model.first_stage_key)
        targets = self.get_conditioning(batch)
        if targets.dim() == 4:
            targets = targets.argmax(dim=1)
        if t is None:
            t = torch.randint(0, self.diffusion_model.num_timesteps, (x.shape[0],), device=self.device).long()
        else:
            t = torch.full(size=(x.shape[0],), fill_value=t, device=self.device).long()
        x_noisy = self.get_x_noisy(x, t)
        logits = self(x_noisy, t)
        loss = F.cross_entropy(logits, targets, reduction='none')
        self.write_logs(loss.detach(), logits.detach(), targets.detach())
        loss = loss.mean()
        return (loss, logits, x_noisy, targets)

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        return loss

    def reset_noise_accs(self):
        self.noisy_acc = {t: {'acc@1': [], 'acc@5': []} for t in range(0, self.diffusion_model.num_timesteps, self.diffusion_model.log_every_t)}

    def on_validation_start(self):
        self.reset_noise_accs()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch)
        for t in self.noisy_acc:
            _, logits, _, targets = self.shared_step(batch, t)
            self.noisy_acc[t]['acc@1'].append(self.compute_top_k(logits, targets, k=1, reduction='mean'))
            self.noisy_acc[t]['acc@5'].append(self.compute_top_k(logits, targets, k=5, reduction='mean'))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            print('Setting up LambdaLR scheduler...')
            scheduler = [{'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
            return ([optimizer], scheduler)
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, N=8, *args, **kwargs):
        log = dict()
        x = self.get_input(batch, self.diffusion_model.first_stage_key)
        log['inputs'] = x
        y = self.get_conditioning(batch)
        if self.label_key == 'class_label':
            y = log_txt_as_img((x.shape[2], x.shape[3]), batch['human_label'])
            log['labels'] = y
        if ismap(y):
            log['labels'] = self.diffusion_model.to_rgb(y)
            for step in range(self.log_steps):
                current_time = step * self.log_time_interval
                _, logits, x_noisy, _ = self.shared_step(batch, t=current_time)
                log[f'inputs@t{current_time}'] = x_noisy
                pred = F.one_hot(logits.argmax(dim=1), num_classes=self.num_classes)
                pred = rearrange(pred, 'b h w c -> b c h w')
                log[f'pred@t{current_time}'] = self.diffusion_model.to_rgb(pred)
        for key in log:
            log[key] = log[key][:N]
        return log

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return len(x.shape) == 4 and x.shape[1] > 3

class LatentDiffusion(DDPM):
    """main class"""

    def __init__(self, first_stage_config, cond_stage_config, num_timesteps_cond=None, cond_stage_key='image', cond_stage_trainable=False, concat_mode=True, cond_stage_forward=None, conditioning_key=None, scale_factor=1.0, scale_by_std=False, *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        super().__init__(*args, conditioning_key=conditioning_key, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.scale_by_std and self.current_epoch == 0 and (self.global_step == 0) and (batch_idx == 0) and (not self.restarted_from_ckpt):
            assert self.scale_factor == 1.0, 'rather not use custom rescaling and std-rescaling simultaneously'
            print('### USING STD-RESCALING ###')
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1.0 / z.flatten().std())
            print(f'setting self.scale_factor to {self.scale_factor}')
            print('### USING STD-RESCALING ###')

    def register_schedule(self, given_betas=None, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == '__is_first_stage__':
                print('Using first stage also as cond stage.')
                self.cond_stage_model = self.first_stage_model
            elif config == '__is_unconditional__':
                print(f'Training {self.__class__.__name__} as an unconditional model.')
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device), force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)
        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params['clip_min_weight'], self.split_input_params['clip_max_weight'])
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)
        if self.split_input_params['tie_braker']:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting, self.split_input_params['clip_min_tie_weight'], self.split_input_params['clip_max_tie_weight'])
            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1
        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)
            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))
        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf), dilation=1, padding=0, stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)
            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))
        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)
            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df), dilation=1, padding=0, stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)
            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))
        else:
            raise NotImplementedError
        return (fold, unfold, normalization, weighting)

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1.0 / self.scale_factor * z
        if hasattr(self, 'split_input_params'):
            if self.split_input_params['patch_distributed_vq']:
                ks = self.split_input_params['ks']
                stride = self.split_input_params['stride']
                uf = self.split_input_params['vqf']
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print('reducing Kernel')
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print('reducing stride')
                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)
                z = unfold(z)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize) for i in range(z.shape[-1])]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]
                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                o = o.view((o.shape[0], -1, o.shape[-1]))
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            elif isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
        elif isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1.0 / self.scale_factor * z
        if hasattr(self, 'split_input_params'):
            if self.split_input_params['patch_distributed_vq']:
                ks = self.split_input_params['ks']
                stride = self.split_input_params['stride']
                uf = self.split_input_params['vqf']
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print('reducing Kernel')
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print('reducing stride')
                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)
                z = unfold(z)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i], force_not_quantize=predict_cids or force_not_quantize) for i in range(z.shape[-1])]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]
                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                o = o.view((o.shape[0], -1, o.shape[-1]))
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            elif isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)
        elif isinstance(self.first_stage_model, VQModelInterface):
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        else:
            return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, 'split_input_params'):
            if self.split_input_params['patch_distributed_vq']:
                ks = self.split_input_params['ks']
                stride = self.split_input_params['stride']
                df = self.split_input_params['vqf']
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print('reducing Kernel')
                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print('reducing stride')
                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
                output_list = [self.first_stage_model.encode(z[:, :, :, :, i]) for i in range(z.shape[-1])]
                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                o = o.view((o.shape[0], -1, o.shape[-1]))
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded
            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):

        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return (x0, y0, w, h)
        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        if hasattr(self, 'split_input_params'):
            assert len(cond) == 1
            assert not return_ids
            ks = self.split_input_params['ks']
            stride = self.split_input_params['stride']
            h, w = x_noisy.shape[-2:]
            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)
            z = unfold(x_noisy)
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]
            if self.cond_stage_key in ['image', 'LR_image', 'segmentation', 'bbox_img'] and self.model.conditioning_key:
                c_key = next(iter(cond.keys()))
                c = next(iter(cond.values()))
                assert len(c) == 1
                c = c[0]
                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))
                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]
            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** num_downs
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w, rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h) for patch_nr in range(z.shape[-1])]
                patch_limits = [(x_tl, y_tl, rescale_latent * ks[0] / full_img_w, rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device) for bbox in patch_limits]
                print(patch_limits_tknzd[0].shape)
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)
                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)
                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]
            else:
                cond_list = [cond for i in range(z.shape[-1])]
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0], tuple)
            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            o = o.view((o.shape[0], -1, o.shape[-1]))
            x_recon = fold(o) / normalization
        else:
            x_recon = self.model(x_noisy, t, **cond)
        if isinstance(x_recon, tuple) and (not return_ids):
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        if self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'eps':
            target = noise
        else:
            raise NotImplementedError()
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{prefix}/loss': loss})
        return (loss, loss_dict)

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)
        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if return_codebook_ids:
            model_out, logits = model_out
        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return (model_mean, posterior_variance, posterior_log_variance, logits)
        elif return_x0:
            return (model_mean, posterior_variance, posterior_log_variance, x_recon)
        else:
            return (model_mean, posterior_variance, posterior_log_variance)

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_codebook_ids=False, quantize_denoised=False, return_x0=False, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None):
        b, *_, device = (*x.shape, x.device)
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_codebook_ids=return_codebook_ids, quantize_denoised=quantize_denoised, return_x0=return_x0, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning('Support dropped.')
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        if return_codebook_ids:
            return (model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1))
        if return_x0:
            return (model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0)
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False, img_callback=None, mask=None, x0=None, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None, log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation', total=timesteps) if verbose else reversed(range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img, x0_partial = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised, return_x0=True, temperature=temperature[i], noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return (img, intermediates)

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))
        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        if return_intermediates:
            return (img, intermediates)
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, verbose=True, timesteps=None, quantize_denoised=False, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond, shape, return_intermediates=return_intermediates, x_T=x_T, verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised, mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)
        return (samples, intermediates)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.0, return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True, **kwargs):
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True, bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log['inputs'] = x
        log['reconstruction'] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, 'decode'):
                xc = self.cond_stage_model.decode(c)
                log['conditioning'] = xc
            elif self.cond_stage_key in ['caption']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch['caption'])
                log['conditioning'] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch['human_label'])
                log['conditioning'] = xc
            elif isimage(xc):
                log['conditioning'] = xc
            if ismap(xc):
                log['original_conditioning'] = self.to_rgb(xc)
        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log['diffusion_row'] = diffusion_grid
        if sample:
            with self.ema_scope('Plotting'):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log['samples'] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log['denoise_row'] = denoise_grid
            if quantize_denoised and (not isinstance(self.first_stage_model, AutoencoderKL)) and (not isinstance(self.first_stage_model, IdentityFirstStage)):
                with self.ema_scope('Plotting Quantized Denoised'):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log['samples_x0_quantized'] = x_samples
            if inpaint:
                b, h, w = (z.shape[0], z.shape[2], z.shape[3])
                mask = torch.ones(N, h, w).to(self.device)
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.0
                mask = mask[:, None, ...]
                with self.ema_scope('Plotting Inpaint'):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta, ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log['samples_inpainting'] = x_samples
                log['mask'] = mask
                with self.ema_scope('Plotting Outpaint'):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta, ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log['samples_outpainting'] = x_samples
        if plot_progressive_rows:
            with self.ema_scope('Plotting Progressives'):
                img, progressives = self.progressive_denoising(c, shape=(self.channels, self.image_size, self.image_size), batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc='Progressive Generation')
            log['progressive_row'] = prog_row
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f'{self.__class__.__name__}: Also optimizing conditioner params!')
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print('Setting up LambdaLR scheduler...')
            scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
            return ([opt], scheduler)
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, 'colorize'):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return len(x.shape) == 4 and (x.shape[1] == 3 or x.shape[1] == 1)

