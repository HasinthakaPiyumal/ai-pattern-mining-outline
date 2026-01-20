# Cluster 150

class UE4BRDF:

    def __init__(self, base_color, metallic, roughness, specular):
        self.base_color = torch.tensor(base_color, dtype=torch.float64).cuda()
        self.base_color = srgb_inv_gamma_correction_torch(self.base_color)
        self.metallic = torch.tensor(metallic, dtype=torch.float64).cuda()
        self.roughness = torch.tensor(roughness, dtype=torch.float64).cuda().clamp(min=0.0001)
        self.specular = torch.tensor(specular, dtype=torch.float64).cuda()
        self.c_diffuse = (1 - self.metallic) * self.base_color
        self.c_specular = (1 - self.metallic) * self.specular + self.metallic * self.base_color

    def lambertian_diffuse(self):
        return self.c_diffuse / np.pi

    def fresnel_schlick(self, h_dot_v):
        return self.c_specular + (1 - self.c_specular) * torch.pow(1 - h_dot_v, 5).repeat_interleave(3, dim=-1)

    def smith_g1(self, n_dot_v):
        k = pow(self.roughness + 1, 2) / 8
        return n_dot_v / (n_dot_v * (1 - k) + k)

    def geometry(self, n_dot_v, n_dot_l):
        return self.smith_g1(n_dot_v) * self.smith_g1(n_dot_l)

    def normal_distribution(self, n, h):
        roughness = pow(self.roughness, 2)
        n_cross_h = torch.cross(n, h, dim=1)
        n_dot_h = torch.einsum('ij,ij->i', n, h).unsqueeze(-1)
        a = n_dot_h * roughness
        k = roughness / (torch.einsum('ij,ij->i', n_cross_h, n_cross_h).unsqueeze(-1) + a * a)
        d = k * k * (1 / np.pi)
        return d

    def evaluate(self, normal, light_dir, view_dir):
        n_dot_l = torch.clamp(torch.dot(normal, light_dir), min=1e-05)
        n_dot_v = torch.clamp(torch.dot(normal, view_dir), min=1e-05)
        half_dir = (light_dir + view_dir) / torch.norm(light_dir + view_dir)
        h_dot_v = torch.clamp(torch.dot(half_dir, view_dir), min=1e-05)
        n_dot_h = torch.clamp(torch.dot(normal, half_dir), min=1e-05)
        diffuse_term = self.lambertian_diffuse()
        specular_term = self.fresnel_schlick(h_dot_v) * self.geometry(n_dot_l, n_dot_v) * self.normal_distribution(normal, half_dir) / (4 * n_dot_l * n_dot_v)
        return diffuse_term + specular_term

    def evaluate_parallel(self, normal, light_dir, view_dir):
        """
        Args:
            normal: [num_samples, 3]
            light_dir: [num_samples, 3]
            view_dir: [num_samples]
        """
        normal = normal.double()
        light_dir = light_dir.double()
        view_dir = view_dir.double()
        num_samples = normal.shape[0]
        n_dot_l = torch.clamp(torch.einsum('ij,ij->i', normal, light_dir).unsqueeze(-1), min=1e-05)
        n_dot_v = torch.clamp(torch.einsum('ij,ij->i', normal, view_dir).unsqueeze(-1), min=1e-05)
        half_dir = (light_dir + view_dir) / torch.norm(light_dir + view_dir, p=2, dim=1, keepdim=True)
        h_dot_v = torch.clamp(torch.einsum('ij,ij->i', half_dir, view_dir).unsqueeze(-1), min=1e-05)
        diffuse_term = self.lambertian_diffuse().expand(num_samples, 3)
        specular_term = self.fresnel_schlick(h_dot_v) * self.geometry(n_dot_l, n_dot_v) * self.normal_distribution(normal, half_dir) / (4 * n_dot_l * n_dot_v)
        brdf = diffuse_term + specular_term
        return brdf.float()

def srgb_inv_gamma_correction_torch(gamma_corrected_image):
    gamma_corrected_image = torch.clamp(gamma_corrected_image, 0, 1)
    linear_image = torch.where(gamma_corrected_image <= 0.04045, gamma_corrected_image / 12.92, ((gamma_corrected_image + 0.055) / 1.055) ** 2.4)
    linear_image = torch.clamp(linear_image, 0, 1)
    return linear_image

class SkyModel(pl.LightningModule):

    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        downsample = hypes['dataset']['downsample']
        self.sky_H = hypes['dataset']['image_H'] // downsample // 2
        self.sky_W = hypes['dataset']['image_W'] // downsample
        self.teacher_prob = hypes['model']['teacher_prob']
        self.env_template = EnvironmentMap(self.sky_H, 'skylatlong')
        world_coord = self.env_template.worldCoordinates()
        self.pos_encoding = torch.from_numpy(np.stack([world_coord[0], world_coord[1], world_coord[2]], axis=-1))
        self.pos_encoding = self.pos_encoding.to('cuda')
        self.input_inv_gamma = hypes['model']['input_inv_gamma']
        self.input_add_pe = hypes['model']['input_add_pe']
        self.encoder_outdim = hypes['model']['ldr_encoder']['args']['layer_channels'][-1]
        self.feat_down = reduce(lambda x, y: x * y, hypes['model']['ldr_encoder']['args']['strides'])
        self.save_hyperparameters()
        self.ldr_encoder = build_module(hypes['model']['ldr_encoder'])
        self.shared_mlp = build_module(hypes['model']['shared_mlp'])
        self.latent_mlp = build_module(hypes['model']['latent_mlp'])
        self.latent_mlp_recon = build_module(hypes['model']['latent_mlp_recon'])
        self.peak_dir_mlp = build_module(hypes['model']['peak_dir_mlp'])
        self.peak_int_mlp = build_module(hypes['model']['peak_int_mlp'])
        self.ldr_decoder = nn.Sequential(build_module(hypes['model']['ldr_decoder']), nn.Sigmoid())
        self.hdr_decoder = build_module(hypes['model']['hdr_decoder'])
        self.ldr_recon_loss = build_loss(hypes['loss']['ldr_recon_loss'])
        self.hdr_recon_loss = build_loss(hypes['loss']['hdr_recon_loss'])
        self.peak_int_loss = build_loss(hypes['loss']['peak_int_loss'])
        self.peak_dir_loss = build_loss(hypes['loss']['peak_dir_loss'])
        self.fix_modules = hypes['model'].get('fix_modules', [])
        self.on_train_epoch_start()

    def encode_forward(self, x):
        """
        Encode LDR panorama to sky vector: 
            1) peak dir 
            2) peak int 
            3) latent vector
            where 1) and 2) can cat together
        
        deep vector -> shared vector -->    latent vector      --> recon deep vector 
                                     |
                                     .->  peak int/dir vector

        Should we add explicit inv gamma to the input?
        """
        if self.input_inv_gamma:
            x = srgb_inv_gamma_correction_torch(x)
        if self.input_add_pe:
            x = x + self.pos_encoding.permute(2, 0, 1)
        deep_feature = self.ldr_encoder(x)
        deep_vector = deep_feature.permute(0, 2, 3, 1).flatten(1)
        shared_vector = self.shared_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(shared_vector)
        peak_int_vector = self.peak_int_mlp(shared_vector)
        latent_vector = self.latent_mlp(shared_vector)
        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True)
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)
        return (peak_vector, latent_vector)

    def decode_forward(self, latent_vector, peak_vector, peak_vector_gt):
        use_gt_peak = False
        if self.training and np.random.rand() < self.teacher_prob:
            use_gt_peak = True
            peak_vector = peak_vector_gt
        B = peak_vector.shape[0]
        peak_dir_encoding, peak_int_encoding = self.build_peak_map(peak_vector)
        decoder_input = torch.cat([peak_dir_encoding, peak_int_encoding, self.pos_encoding.expand(B, -1, -1, -1)], dim=-1)
        decoder_input = decoder_input.permute(0, 3, 1, 2)
        recon_deep_vector = self.latent_mlp_recon(latent_vector)
        recon_deep_feature = recon_deep_vector.view(B, self.sky_H // self.feat_down, self.sky_W // self.feat_down, self.encoder_outdim).permute(0, 3, 1, 2)
        ldr_skypano_recon = self.ldr_decoder(recon_deep_feature)
        hdr_skypano_recon = self.hdr_decoder(decoder_input, recon_deep_feature)
        return (hdr_skypano_recon, ldr_skypano_recon, use_gt_peak)

    def build_peak_map(self, peak_vector):
        """
        Args:
            peak_vector: [B, 6]
                3 for peak dir, 3 for peak intensity

        Returns:
            peak encoding map: [B, 4, H, W]
                1 for peak dir using spherical gaussian lobe, 3 for peak intensity
        """
        dir_vector = peak_vector[..., :3]
        int_vector = peak_vector[..., 3:]
        dir_vector_expand = dir_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1)
        peak_dir_encoding = torch.exp(100 * (torch.einsum('nhwc,nhwc->nhw', dir_vector_expand, self.pos_encoding.expand(dir_vector_expand.shape)) - 1)).unsqueeze(-1)
        sun_mask = torch.gt(peak_dir_encoding, 0.9).expand(-1, -1, -1, 3)
        int_vector_expand = int_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1)
        peak_int_encoding = torch.where(sun_mask, int_vector_expand, 0)
        return (peak_dir_encoding, peak_int_encoding)

    def on_train_epoch_start(self):
        print(f'Module fixed in training: {self.fix_modules}.')
        for module in self.fix_modules:
            for p in eval(f'self.{module}').parameters():
                p.requires_grad_(False)
            eval(f'self.{module}').eval()

    def training_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_gt)
        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[..., :3], peak_vector_gt[..., :3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[..., 3:], peak_vector_gt[..., 3:])
        loss = hdr_recon_loss + peak_dir_loss + peak_int_loss + ldr_recon_loss
        self.log('train_loss', loss)
        self.log('hdr_recon_loss', hdr_recon_loss)
        self.log('ldr_recon_loss', ldr_recon_loss)
        self.log('peak_dir_loss', peak_dir_loss)
        self.log('peak_int_loss', peak_int_loss)
        log_info = f'|| loss: {loss:.3f} || hdr_recon_loss: {hdr_recon_loss:.3f}  || ldr_recon_loss: {ldr_recon_loss:.3f} || peak_dir_loss: {peak_dir_loss:.3f} ' + f'|| peak_int_loss: {peak_int_loss:.3f}'
        print(log_info)
        return loss

    def validation_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred)
        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        loss = hdr_recon_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred)
        print(f'{batch_idx:0>3} \n                  HDRI Peak Intensity:\t\t {hdr_skypano_pred[0].flatten(1, 2).max(dim=-1)[0]} \n                  Peak Intensity Vector:\t {peak_vector_pred[0][3:]} \n                  Ground Truth Peak Intensity:\t {peak_vector_gt[0][3:]}')
        return_dict = {'ldr_skypano_input': ldr_skypano.permute(0, 2, 3, 1), 'ldr_skypano_pred': ldr_skypano_recon.permute(0, 2, 3, 1), 'hdr_skypano_gt': hdr_skypano_gt.permute(0, 2, 3, 1), 'hdr_skypano_pred': hdr_skypano_pred.permute(0, 2, 3, 1), 'batch_idx': batch_idx}
        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypes['lr_schedule']['init_lr'])
        lr_scheduler = StepLR(optimizer=optimizer, step_size=self.hypes['lr_schedule']['decay_per_epoch'], gamma=self.hypes['lr_schedule']['decay_rate'])
        return ([optimizer], [lr_scheduler])

class SkyModelEnhanced(pl.LightningModule):

    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        downsample = hypes['dataset']['downsample']
        self.sky_H = hypes['dataset']['image_H'] // downsample // 2
        self.sky_W = hypes['dataset']['image_W'] // downsample
        self.teacher_prob = hypes['model']['teacher_prob']
        self.env_template = EnvironmentMap(self.sky_H, 'skylatlong')
        world_coord = self.env_template.worldCoordinates()
        self.pos_encoding = torch.from_numpy(np.stack([world_coord[0], world_coord[1], world_coord[2]], axis=-1))
        self.pos_encoding = self.pos_encoding.to('cuda')
        self.input_inv_gamma = hypes['model']['input_inv_gamma']
        self.input_add_pe = hypes['model']['input_add_pe']
        self.encoder_outdim = hypes['model']['ldr_encoder']['args']['layer_channels'][-1]
        self.feat_down = reduce(lambda x, y: x * y, hypes['model']['ldr_encoder']['args']['strides'])
        self.sum_lobe_thres = hypes['model'].get('sum_lobe_thres', 0.9)
        self.save_hyperparameters()
        self.ldr_encoder = build_module(hypes['model']['ldr_encoder'])
        self.shared_mlp = build_module(hypes['model']['shared_mlp'])
        self.latent_mlp = build_module(hypes['model']['latent_mlp'])
        self.latent_mlp_recon = build_module(hypes['model']['latent_mlp_recon'])
        self.peak_dir_mlp = build_module(hypes['model']['peak_dir_mlp'])
        self.peak_int_mlp = build_module(hypes['model']['peak_int_mlp'])
        self.ldr_decoder = nn.Sequential(build_module(hypes['model']['ldr_decoder']), nn.Sigmoid())
        self.hdr_decoder = build_module(hypes['model']['hdr_decoder'])
        self.ldr_recon_loss = build_loss(hypes['loss']['ldr_recon_loss'])
        self.hdr_recon_loss = build_loss(hypes['loss']['hdr_recon_loss'])
        self.peak_int_loss = build_loss(hypes['loss']['peak_int_loss'])
        self.peak_dir_loss = build_loss(hypes['loss']['peak_dir_loss'])
        self.fix_modules = hypes['model'].get('fix_modules', [])
        self.on_train_epoch_start()

    def encode_forward(self, x):
        """
        Encode LDR panorama to sky vector: 
            1) peak dir 
            2) peak int 
            3) latent vector
            where 1) and 2) can cat together
        
        deep vector -> shared vector -->    latent vector      --> recon deep vector 
                                     |
                                     .->  peak int/dir vector

        Should we add explicit inv gamma to the input?
        """
        if self.input_inv_gamma:
            x = srgb_inv_gamma_correction_torch(x)
        if self.input_add_pe:
            x = x + self.pos_encoding.permute(2, 0, 1)
        deep_feature = self.ldr_encoder(x)
        deep_vector = deep_feature.permute(0, 2, 3, 1).flatten(1)
        shared_vector = self.shared_mlp(deep_vector)
        peak_dir_vector = self.peak_dir_mlp(shared_vector)
        peak_int_vector = self.peak_int_mlp(shared_vector)
        latent_vector = self.latent_mlp(shared_vector)
        peak_dir_vector = peak_dir_vector / peak_dir_vector.norm(dim=1, keepdim=True)
        peak_vector = torch.cat([peak_dir_vector, peak_int_vector], dim=-1)
        return (peak_vector, latent_vector)

    def decode_forward(self, latent_vector, peak_vector, peak_vector_gt):
        use_gt_peak = False
        if self.training and np.random.rand() < self.teacher_prob:
            use_gt_peak = True
            peak_vector = peak_vector_gt
        B = peak_vector.shape[0]
        peak_dir_encoding, peak_int_encoding, sum_mask = self.build_peak_map(peak_vector)
        decoder_input = torch.cat([peak_dir_encoding, peak_int_encoding, self.pos_encoding.expand(B, -1, -1, -1)], dim=-1)
        decoder_input = decoder_input.permute(0, 3, 1, 2)
        recon_deep_vector = self.latent_mlp_recon(latent_vector)
        recon_deep_feature = recon_deep_vector.view(B, self.sky_H // self.feat_down, self.sky_W // self.feat_down, self.encoder_outdim).permute(0, 3, 1, 2)
        ldr_skypano_recon = self.ldr_decoder(recon_deep_feature)
        hdr_skypano_recon = self.hdr_decoder(decoder_input, recon_deep_feature)
        sum_mask = sum_mask.permute(0, 3, 1, 2)
        sun_peak_map = peak_dir_encoding.permute(0, 3, 1, 2) * peak_int_encoding.permute(0, 3, 1, 2)
        hdr_skypano_recon = torch.where(sum_mask, sun_peak_map, hdr_skypano_recon)
        return (hdr_skypano_recon, ldr_skypano_recon, use_gt_peak)

    def build_peak_map(self, peak_vector):
        """
        Args:
            peak_vector : [B, 6]
                3 for peak dir, 3 for peak intensity

        Returns:
            peak encoding map : [B, H, W, 4]
                1 for peak dir using spherical gaussian lobe, 3 for peak intensity

            sum_mask :[B, H, W, 3]
        """
        dir_vector = peak_vector[..., :3]
        int_vector = peak_vector[..., 3:]
        dir_vector_expand = dir_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1)
        peak_dir_cosine = torch.einsum('nhwc,nhwc->nhw', dir_vector_expand, self.pos_encoding.expand(dir_vector_expand.shape)).unsqueeze(-1)
        peak_dir_encoding = torch.exp(100 * (peak_dir_cosine - 1))
        sun_mask = torch.gt(peak_dir_encoding, self.sum_lobe_thres).expand(-1, -1, -1, 3)
        int_vector_expand = int_vector.unsqueeze(1).unsqueeze(1).expand(-1, self.sky_H, self.sky_W, -1)
        peak_int_encoding = torch.where(sun_mask, int_vector_expand, 0)
        return (peak_dir_encoding, peak_int_encoding, sun_mask)

    def on_train_epoch_start(self):
        print(f'Module fixed in training: {self.fix_modules}.')
        for module in self.fix_modules:
            for p in eval(f'self.{module}').parameters():
                p.requires_grad_(False)
            eval(f'self.{module}').eval()

    def training_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_gt)
        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[..., :3], peak_vector_gt[..., :3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[..., 3:], peak_vector_gt[..., 3:])
        loss = hdr_recon_loss + peak_dir_loss + peak_int_loss + ldr_recon_loss
        self.log('train_loss', loss)
        self.log('hdr_recon_loss', hdr_recon_loss)
        self.log('ldr_recon_loss', ldr_recon_loss)
        self.log('peak_dir_loss', peak_dir_loss)
        self.log('peak_int_loss', peak_int_loss)
        log_info = f'|| loss: {loss:.3f} || hdr_recon_loss: {hdr_recon_loss:.3f}  || ldr_recon_loss: {ldr_recon_loss:.3f} || peak_dir_loss: {peak_dir_loss:.3f} ' + f'|| peak_int_loss: {peak_int_loss:.3f}'
        print(log_info)
        return loss

    def validation_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred)
        ldr_recon_loss = self.ldr_recon_loss(ldr_skypano_recon, ldr_skypano)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_skypano_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[..., :3], peak_vector_gt[..., :3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[..., 3:], peak_vector_gt[..., 3:])
        loss = hdr_recon_loss + peak_dir_loss + peak_int_loss + ldr_recon_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        ldr_skypano, hdr_skypano_gt, peak_vector_gt = batch
        peak_vector_pred, latent_vector = self.encode_forward(ldr_skypano)
        hdr_skypano_pred, ldr_skypano_recon, _ = self.decode_forward(latent_vector, peak_vector_pred, peak_vector_pred)
        print(f'{batch_idx:0>3} \n                  HDRI Peak Intensity:\t\t {hdr_skypano_pred[0].flatten(1, 2).max(dim=-1)[0]} \n                  Peak Intensity Vector:\t {peak_vector_pred[0][3:]} \n                  Ground Truth Peak Intensity:\t {peak_vector_gt[0][3:]}')
        return_dict = {'ldr_skypano_input': ldr_skypano.permute(0, 2, 3, 1), 'ldr_skypano_pred': ldr_skypano_recon.permute(0, 2, 3, 1), 'hdr_skypano_gt': hdr_skypano_gt.permute(0, 2, 3, 1), 'hdr_skypano_pred': hdr_skypano_pred.permute(0, 2, 3, 1), 'batch_idx': batch_idx}
        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypes['lr_schedule']['init_lr'])
        lr_scheduler = StepLR(optimizer=optimizer, step_size=self.hypes['lr_schedule']['decay_per_epoch'], gamma=self.hypes['lr_schedule']['decay_rate'])
        return ([optimizer], [lr_scheduler])

