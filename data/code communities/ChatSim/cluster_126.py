# Cluster 126

def hann2d(sz: torch.Tensor, centered=True) -> torch.Tensor:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)

def hann1d(sz: int, centered=True) -> torch.Tensor:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - torch.cos(2 * math.pi / (sz + 1) * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos(2 * math.pi / (sz + 2) * torch.arange(0, sz // 2 + 1).float()))
    return torch.cat([w, w[1:sz - sz // 2].flip((0,))])

def hann2d_bias(sz: torch.Tensor, ctr_point: torch.Tensor, centered=True) -> torch.Tensor:
    """2D cosine window."""
    distance = torch.stack([ctr_point, sz - ctr_point], dim=0)
    max_distance, _ = distance.max(dim=0)
    hann1d_x = hann1d(max_distance[0].item() * 2, centered)
    hann1d_x = hann1d_x[max_distance[0] - distance[0, 0]:max_distance[0] + distance[1, 0]]
    hann1d_y = hann1d(max_distance[1].item() * 2, centered)
    hann1d_y = hann1d_y[max_distance[1] - distance[0, 1]:max_distance[1] + distance[1, 1]]
    return hann1d_y.reshape(1, 1, -1, 1) * hann1d_x.reshape(1, 1, 1, -1)

def hann2d_clipped(sz: torch.Tensor, effective_sz: torch.Tensor, centered=True) -> torch.Tensor:
    """1D clipped cosine window."""
    effective_sz += (effective_sz - sz) % 2
    effective_window = hann1d(effective_sz[0].item(), True).reshape(1, 1, -1, 1) * hann1d(effective_sz[1].item(), True).reshape(1, 1, 1, -1)
    pad = (sz - effective_sz) // 2
    window = F.pad(effective_window, (pad[1].item(), pad[1].item(), pad[0].item(), pad[0].item()), 'replicate')
    if centered:
        return window
    else:
        mid = (sz / 2).int()
        window_shift_lr = torch.cat((window[:, :, :, mid[1]:], window[:, :, :, :mid[1]]), 3)
        return torch.cat((window_shift_lr[:, :, mid[0]:, :], window_shift_lr[:, :, :mid[0], :]), 2)

class OSTrack(BaseTracker):

    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = 'debug'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor, template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            'save all predicted boxes'
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict=None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = search
            out_dict = self.network.forward(template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, '%04d.jpg' % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
        if self.save_all_boxes:
            'save all predictions'
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {'target_bbox': self.state, 'all_boxes': all_boxes_save}
        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = (self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = (self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = ([], [], [])
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1]))
        self.enc_attn_weights = enc_attn_weights

