# Cluster 7

class FaceRestorationHelper(object):
    """Helper for the face restoration pipeline."""

    def __init__(self, upscale_factor, face_size=512):
        self.upscale_factor = upscale_factor
        self.face_size = (face_size, face_size)
        self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941], [337.91089109, 488.38613861], [437.95049505, 493.51485149], [513.58415842, 678.5049505]])
        self.face_template = self.face_template / (1024 // face_size)
        self.similarity_trans = trans.SimilarityTransform()
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.save_png = True

    def init_dlib(self, detection_path, landmark5_path, landmark68_path):
        """Initialize the dlib detectors and predictors."""
        self.face_detector = dlib.cnn_face_detection_model_v1(detection_path)
        self.shape_predictor_5 = dlib.shape_predictor(landmark5_path)
        self.shape_predictor_68 = dlib.shape_predictor(landmark68_path)

    def free_dlib_gpu_memory(self):
        del self.face_detector
        del self.shape_predictor_5
        del self.shape_predictor_68

    def read_input_image(self, img_path):
        self.input_img = dlib.load_rgb_image(img_path)

    def detect_faces(self, img_path, upsample_num_times=1, only_keep_largest=False):
        """
        Args:
            img_path (str): Image path.
            upsample_num_times (int): Upsamples the image before running the
                face detector

        Returns:
            int: Number of detected faces.
        """
        self.read_input_image(img_path)
        det_faces = self.face_detector(self.input_img, upsample_num_times)
        if len(det_faces) == 0:
            print('No face detected. Try to increase upsample_num_times.')
        elif only_keep_largest:
            print('Detect several faces and only keep the largest.')
            face_areas = []
            for i in range(len(det_faces)):
                face_area = (det_faces[i].rect.right() - det_faces[i].rect.left()) * (det_faces[i].rect.bottom() - det_faces[i].rect.top())
                face_areas.append(face_area)
            largest_idx = face_areas.index(max(face_areas))
            self.det_faces = [det_faces[largest_idx]]
        else:
            self.det_faces = det_faces
        return len(self.det_faces)

    def get_face_landmarks_5(self):
        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)
        return len(self.all_landmarks_5)

    def get_face_landmarks_68(self):
        """Get 68 densemarks for cropped images.

        Should only have one face at most in the cropped image.
        """
        num_detected_face = 0
        for idx, face in enumerate(self.cropped_faces):
            det_face = self.face_detector(face, 1)
            if len(det_face) == 0:
                print(f'Cannot find faces in cropped image with index {idx}.')
                self.all_landmarks_68.append(None)
            else:
                if len(det_face) > 1:
                    print('Detect several faces in the cropped face. Use the  largest one. Note that it will also cause overlap during paste_faces_to_input_image.')
                    face_areas = []
                    for i in range(len(det_face)):
                        face_area = (det_face[i].rect.right() - det_face[i].rect.left()) * (det_face[i].rect.bottom() - det_face[i].rect.top())
                        face_areas.append(face_area)
                    largest_idx = face_areas.index(max(face_areas))
                    face_rect = det_face[largest_idx].rect
                else:
                    face_rect = det_face[0].rect
                shape = self.shape_predictor_68(face, face_rect)
                landmark = np.array([[part.x, part.y] for part in shape.parts()])
                self.all_landmarks_68.append(landmark)
                num_detected_face += 1
        return num_detected_face

    def warp_crop_faces(self, save_cropped_path=None, save_inverse_affine_path=None):
        """Get affine matrix, warp and cropped faces.

        Also get inverse affine matrix for post-processing.
        """
        for idx, landmark in enumerate(self.all_landmarks_5):
            self.similarity_trans.estimate(landmark, self.face_template)
            affine_matrix = self.similarity_trans.params[0:2, :]
            self.affine_matrices.append(affine_matrix)
            cropped_face = cv2.warpAffine(self.input_img, affine_matrix, self.face_size)
            self.cropped_faces.append(cropped_face)
            if save_cropped_path is not None:
                path, ext = os.path.splitext(save_cropped_path)
                if self.save_png:
                    save_path = f'{path}_{idx:02d}.png'
                else:
                    save_path = f'{path}_{idx:02d}{ext}'
                imwrite(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), save_path)
            self.similarity_trans.estimate(self.face_template, landmark * self.upscale_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(inverse_affine)
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_faces_to_input_image(self, save_path):
        input_img = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2BGR)
        h, w, _ = input_img.shape
        h_up, w_up = (h * self.upscale_factor, w * self.upscale_factor)
        upsample_img = cv2.resize(input_img, (w_up, h_up))
        assert len(self.restored_faces) == len(self.inverse_affine_matrices), 'length of restored_faces and affine_matrices are different.'
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))
            mask = np.ones((*self.face_size, 3), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            inv_mask_erosion = cv2.erode(inv_mask, np.ones((2 * self.upscale_factor, 2 * self.upscale_factor), np.uint8))
            inv_restored_remove_border = inv_mask_erosion * inv_restored
            total_face_area = np.sum(inv_mask_erosion) // 3
            w_edge = int(total_face_area ** 0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            upsample_img = inv_soft_mask * inv_restored_remove_border + (1 - inv_soft_mask) * upsample_img
        if self.save_png:
            save_path = save_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        imwrite(upsample_img.astype(np.uint8), save_path)

    def clean_all(self):
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]
        loss_dict = OrderedDict()
        l_pix = 0.0
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix
        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = (0, 0)
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test
        cnt = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                    save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            cnt += 1
        current_metric = 0.0
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all((torch.is_tensor(t) for t in tensor)))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

