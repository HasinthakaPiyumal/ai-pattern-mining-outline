# Cluster 147

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def print(*args, **kwargs):
    force = kwargs.pop('force', False)
    if is_master or force:
        builtin_print(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None

    def predicts_segmentation_mask(self):
        return False

    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, info: dict=None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, segmentation), 'Tracking', 1, 'Tracking')

    def transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        if crop_type == 'template':
            crop_sz = torch.Tensor([self.params.template_size, self.params.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        else:
            raise NotImplementedError
        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)
        return template_bbox

    def _init_visdom(self, visdom_info, debug):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        if debug > 0 and visdom_info.get('use_visdom', True):
            try:
                self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'}, visdom_info=visdom_info)
            except:
                time.sleep(0.5)
                print("!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n!!! Start Visdom in a separate terminal window by typing 'visdom' !!!")

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode
            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True
            elif data['key'] == 'n':
                self.next_seq = True

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int=None, display_name: str=None, result_only=False):
        assert run_id is None or isinstance(run_id, int)
        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        init_info = seq.init_info()
        tracker = self.create_tracker(params)
        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        output = {'target_bbox': [], 'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)
        image = self._read_image(seq.frames[0])
        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}
        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'), 'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']
        _store_outputs(out, init_default)
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            start_time = time.time()
            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output
            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)
        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))
        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))
        assert os.path.isfile(videofilepath), 'Invalid param {}'.format(videofilepath)
        ', videofilepath must be a valid videofile'
        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}
        if success is not True:
            print('Read frame from {} failed.'.format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                frame_disp = frame.copy()
                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 1)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame_disp = frame.copy()
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), (0, 255, 0), 5)
            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()
                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 1)
                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
        cap.release()
        cv.destroyAllWindows()
        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))
            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError('type of image_file should be str or list')

def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = (posemb[:, :num_tokens], posemb[0, num_tokens:])
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = (posemb[:, :0], posemb[0])
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    print('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location='cpu')
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
            print('Load pretrained model from: ' + pretrained)
    return model

class BaseBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384
        self.cat_mode = 'direct'
        self.pos_embed_z = None
        self.pos_embed_x = None
        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None
        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]
        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE
        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size), mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3, embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size)
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = search_size
        new_P_H, new_P_W = (H // new_patch_size, W // new_patch_size)
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)
        H, W = template_size
        new_P_H, new_P_W = (H // new_patch_size, W // new_patch_size)
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic', align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=0.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=0.02)
        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-06)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x):
        B, H, W = (x.shape[0], x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed
        z += self.pos_embed_z
        x += self.pos_embed_x
        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed
        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        aux_dict = {'attn': None}
        return (self.norm(x), aux_dict)

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x)
        return (x, aux_dict)

def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE
    if cfg.MODEL.HEAD.TYPE == 'MLP':
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)
        return mlp_head
    elif 'CORNER' in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, 'NUM_CHANNELS', 256)
        print('head channel: %d' % channel)
        if cfg.MODEL.HEAD.TYPE == 'CORNER':
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel, feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == 'CENTER':
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel, feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError('HEAD TYPE %s is not supported.' % cfg.MODEL.HEAD_TYPE)

@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None, mask=None, x0=None, quantize_x0=False, img_callback=None, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, x_T=None, log_every_t=None):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    print(f'Sampling with eta = {eta}; steps: {steps}')
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback, normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta, mask=mask, x0=x0, temperature=temperature, verbose=False, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, x_T=x_T)
    return (samples, intermediates)

@torch.no_grad()
def make_convolutional_sample(batch, model, mode='vanilla', custom_steps=None, eta=1.0, swap_mode=False, masked=False, invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000, resize_enabled=False, custom_shape=None, temperature=1.0, noise_dropout=0.0, corrector=None, corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True, ddim_use_x0_pred=False):
    log = dict()
    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key, return_first_stage_outputs=True, force_c_encode=not (hasattr(model, 'split_input_params') and model.cond_stage_key == 'coordinates_bbox'), return_original_cond=True)
    log_every_t = 1 if save_intermediate_vid else None
    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f'Generating {custom_shape[0]} samples of shape {custom_shape[1:]}')
    z0 = None
    log['input'] = x
    log['reconstruction'] = xrec
    if ismap(xc):
        log['original_conditioning'] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)
    else:
        log['original_conditioning'] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key == 'class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]
    with model.ema_scope('Plotting'):
        t0 = time.time()
        img_cb = None
        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape, eta=eta, quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0, temperature=temperature, noise_dropout=noise_dropout, score_corrector=corrector, corrector_kwargs=corrector_kwargs, x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()
        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]
    x_sample = model.decode_first_stage(sample)
    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log['sample_noquant'] = x_sample_noquant
        log['sample_diff'] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass
    log['sample'] = x_sample
    log['time'] = t1 - t0
    return log

class SetupCallback(Callback):

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print('Summoning checkpoint.')
            ckpt_path = os.path.join(self.ckptdir, 'last.ckpt')
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            if 'callbacks' in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print('Project config')
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, '{}-project.yaml'.format(self.now)))
            print('Lightning config')
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}), os.path.join(self.cfgdir, '{}-lightning.yaml'.format(self.now)))
        elif not self.resume and os.path.exists(self.logdir):
            dst, name = os.path.split(self.logdir)
            dst = os.path.join(dst, 'child_runs', name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            try:
                os.rename(self.logdir, dst)
            except FileNotFoundError:
                pass

class ImageLogger(Callback):

    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {pl.loggers.TestTubeLogger: self._testtube}
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0
            tag = f'{split}/{k}'
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, 'images', split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:06}_b-{:06}.png'.format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.check_frequency(check_idx) and hasattr(pl_module, 'log_images') and callable(pl_module.log_images) and (self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)
            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq == 0 or check_idx in self.log_steps) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split='val')
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted((k for k in vars(args) if getattr(opt, k) != getattr(args, k)))

def melk(*args, **kwargs):
    if trainer.global_rank == 0:
        print('Summoning checkpoint.')
        ckpt_path = os.path.join(ckptdir, 'last.ckpt')
        trainer.save_checkpoint(ckpt_path)

def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)
    model.cuda()
    model.eval()
    return model

def logs2pil(logs, keys=['sample']):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f'Unknown format for key {k}. ')
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == 'RGB':
        x = x.convert('RGB')
    return x

@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0):
    log = dict()
    shape = [batch_size, model.model.diffusion_model.in_channels, model.model.diffusion_model.image_size, model.model.diffusion_model.image_size]
    with model.ema_scope('Plotting'):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape, make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape, eta=eta)
        t1 = time.time()
    x_sample = model.decode_first_stage(sample)
    log['sample'] = x_sample
    log['time'] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log['throughput']}')
    return log

@torch.no_grad()
def convsample(model, shape, return_intermediates=True, verbose=True, make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape, return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(None, shape, verbose=True)

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')
    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    if model.cond_stage_model is None:
        all_images = []
        print(f'Running unconditional sampling for {n_samples} samples')
        for _ in trange(n_samples // batch_size, desc='Sampling Batches (unconditional)'):
            logs = make_convolutional_sample(model, batch_size=batch_size, vanilla=vanilla, custom_steps=custom_steps, eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key='sample')
            all_images.extend([custom_to_np(logs['sample'])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = 'x'.join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f'{shape_str}-samples.npz')
        np.savez(nppath, all_img)
    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')
    print(f'sampling of {n_saved} images finished in {(time.time() - tstart) / 60.0:.2f} minutes.')

def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def save_logs(logs, path, n_saved=0, key='sample', np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f'{key}_{n_saved:06}.png')
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = 'x'.join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f'{n_saved}-{shape_str}-samples.npz')
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd['global_step']}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)
    model.cuda()
    model.eval()
    return model

class Searcher(object):

    def __init__(self, database, retriever_version='ViT-L/14'):
        assert database in DATABASES
        self.database_name = database
        self.searcher_savedir = f'data/rdm/searchers/{self.database_name}'
        self.database_path = f'data/rdm/retrieval_databases/{self.database_name}'
        self.retriever = self.load_retriever(version=retriever_version)
        self.database = {'embedding': [], 'img_id': [], 'patch_coords': []}
        self.load_database()
        self.load_searcher()

    def train_searcher(self, k, metric='dot_product', searcher_savedir=None):
        print('Start training searcher')
        searcher = scann.scann_ops_pybind.builder(self.database['embedding'] / np.linalg.norm(self.database['embedding'], axis=1)[:, np.newaxis], k, metric)
        self.searcher = searcher.score_brute_force().build()
        print('Finish training searcher')
        if searcher_savedir is not None:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_single_file(self, saved_embeddings):
        compressed = np.load(saved_embeddings)
        self.database = {key: compressed[key] for key in compressed.files}
        print('Finished loading of clip embeddings.')

    def load_multi_files(self, data_archive):
        out_data = {key: [] for key in self.database}
        for d in tqdm(data_archive, desc=f'Loading datapool from {len(data_archive)} individual files.'):
            for key in d.files:
                out_data[key].append(d[key])
        return out_data

    def load_database(self):
        print(f'Load saved patch embedding from "{self.database_path}"')
        file_content = glob.glob(os.path.join(self.database_path, '*.npz'))
        if len(file_content) == 1:
            self.load_single_file(file_content[0])
        elif len(file_content) > 1:
            data = [np.load(f) for f in file_content]
            prefetched_data = parallel_data_prefetch(self.load_multi_files, data, n_proc=min(len(data), cpu_count()), target_data_type='dict')
            self.database = {key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0] for key in self.database}
        else:
            raise ValueError(f'No npz-files in specified path "{self.database_path}" is this directory existing?')
        print(f'Finished loading of retrieval database of length {self.database['embedding'].shape[0]}.')

    def load_retriever(self, version='ViT-L/14'):
        model = FrozenClipImageEmbedder(model=version)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def load_searcher(self):
        print(f'load searcher for database {self.database_name} from {self.searcher_savedir}')
        self.searcher = scann.scann_ops_pybind.load_searcher(self.searcher_savedir)
        print('Finished loading searcher.')

    def search(self, x, k):
        if self.searcher is None and self.database['embedding'].shape[0] < 20000.0:
            self.train_searcher(k)
        assert self.searcher is not None, 'Cannot search with uninitialized searcher'
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(x.shape) == 3:
            x = x[:, 0]
        query_embeddings = x / np.linalg.norm(x, axis=1)[:, np.newaxis]
        start = time.time()
        nns, distances = self.searcher.search_batched(query_embeddings, final_num_neighbors=k)
        end = time.time()
        out_embeddings = self.database['embedding'][nns]
        out_img_ids = self.database['img_id'][nns]
        out_pc = self.database['patch_coords'][nns]
        out = {'nn_embeddings': out_embeddings / np.linalg.norm(out_embeddings, axis=-1)[..., np.newaxis], 'img_ids': out_img_ids, 'patch_coords': out_pc, 'queries': x, 'exec_time': end - start, 'nns': nns, 'q_embeddings': query_embeddings}
        return out

    def __call__(self, x, n):
        return self.search(x, n)

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f'current step: {n}, recent lr-multiplier: {self.last_lr}')
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)

class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f'current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}')
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)

class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f'current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}')
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / self.cycle_lengths[cycle]
            self.last_f = f
            return f

def count_params(model, verbose=False):
    total_params = sum((p.numel() for p in model.parameters()))
    if verbose:
        print(f'{model.__class__.__name__} has {total_params * 1e-06:.2f} M params.')
    return total_params

def parallel_data_prefetch(func: callable, data, n_proc, target_data_type='ndarray', cpu_intensive=True, use_worker_id=False):
    if isinstance(data, np.ndarray) and target_data_type == 'list':
        raise ValueError('list expected but function got ndarray.')
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.')
            data = list(data.values())
        if target_data_type == 'ndarray':
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(f'The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}.')
    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    if target_data_type == 'ndarray':
        arguments = [[func, Q, part, i, use_worker_id] for i, part in enumerate(np.array_split(data, n_proc))]
    else:
        step = int(len(data) / n_proc + 1) if len(data) % n_proc != 0 else int(len(data) / n_proc)
        arguments = [[func, Q, part, i, use_worker_id] for i, part in enumerate([data[i:i + step] for i in range(0, len(data), step)])]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]
    print(f'Start prefetching...')
    import time
    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()
        k = 0
        while k < n_proc:
            res = Q.get()
            if res == 'Done':
                k += 1
            else:
                gather_res[res[0]] = res[1]
    except Exception as e:
        print('Exception: ', e)
        for p in processes:
            p.terminate()
        raise e
    finally:
        for p in processes:
            p.join()
        print(f'Prefetching complete. [{time.time() - start} sec.]')
    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res

class VQLPIPSWithDiscriminator(nn.Module):

    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, disc_loss='hinge', n_classes=None, perceptual_loss='lpips', pixel_loss='l1'):
        super().__init__()
        assert disc_loss in ['hinge', 'vanilla']
        assert perceptual_loss in ['lpips', 'clips', 'dists']
        assert pixel_loss in ['l1', 'l2']
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        if perceptual_loss == 'lpips':
            print(f'{self.__class__.__name__}: Running with LPIPS.')
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f'Unknown perceptual loss: >> {perceptual_loss} <<')
        self.perceptual_weight = perceptual_weight
        if pixel_loss == 'l1':
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm, ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f'VQLPIPSWithDiscriminator running with {disc_loss} loss.')
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 0.0001)
        d_weight = torch.clamp(d_weight, 0.0, 10000.0).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, cond=None, split='train', predicted_indices=None):
        if not exists(codebook_loss):
            codebook_loss = torch.tensor([0.0]).to(inputs.device)
        rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            log = {'{}/total_loss'.format(split): loss.clone().detach().mean(), '{}/quant_loss'.format(split): codebook_loss.detach().mean(), '{}/nll_loss'.format(split): nll_loss.detach().mean(), '{}/rec_loss'.format(split): rec_loss.detach().mean(), '{}/p_loss'.format(split): p_loss.detach().mean(), '{}/d_weight'.format(split): d_weight.detach(), '{}/disc_factor'.format(split): torch.tensor(disc_factor), '{}/g_loss'.format(split): g_loss.detach().mean()}
            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f'{split}/perplexity'] = perplexity
                log[f'{split}/cluster_usage'] = cluster_usage
            return (loss, log)
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            log = {'{}/disc_loss'.format(split): d_loss.clone().detach().mean(), '{}/logits_real'.format(split): logits_real.detach().mean(), '{}/logits_fake'.format(split): logits_fake.detach().mean()}
            return (d_loss, log)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

class Upsampler(nn.Module):

    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size // in_size)) + 1
        factor_up = 1.0 + out_size % in_size
        print(f'Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}')
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2 * in_channels, out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2, attn_resolutions=[], in_channels=None, ch=in_channels, ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x

class Resize(nn.Module):

    def __init__(self, in_channels=None, learned=False, mode='bilinear'):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f'Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode')
            raise NotImplementedError()
            assert in_channels is not None
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor == 1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult: list, in_channels, pretrained_model: nn.Module=None, reshape=False, n_channels=None, dropout=0.0, pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)
        self.do_reshape = reshape
        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch
        self.proj_norm = Normalize(in_channels, num_groups=in_channels // 2)
        self.proj = nn.Conv2d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in, out_channels=m * n_channels, dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))
        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)

    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_with_pretrained(self, x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return c

    def forward(self, x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)
        for submodel, downmodel in zip(self.model, self.downsampler):
            z = submodel(z, temb=None)
            z = downmodel(z)
        if self.do_reshape:
            z = rearrange(z, 'b c h w -> b (h w) c')
        return z

def instantiate_from_config(config):
    if not 'target' in config:
        if config == '__is_first_stage__':
            return None
        elif config == '__is_unconditional__':
            return None
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}

class SpatialRescaler(nn.Module):

    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class VQModel(pl.LightningModule):

    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[], image_key='image', colorize_nlabels=None, monitor=None, batch_resize_range=None, scheduler_config=None, lr_g_factor=1.0, remap=None, sane_index_shape=False, use_ema=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f'{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.')
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f'Keeping EMAs of {len(list(self.model_ema.buffers()))}.')
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys')
        if len(missing) > 0:
            print(f'Missing Keys: {missing}')
            print(f'Unexpected Keys: {unexpected}')

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return (quant, emb_loss, info)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return (dec, diff, ind)
        return (dec, diff)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode='bicubic')
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split='train', predicted_indices=ind)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss
        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split='train')
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix='_ema')
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=''):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split='val' + suffix, predicted_indices=ind)
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split='val' + suffix, predicted_indices=ind)
        rec_loss = log_dict_ae[f'val{suffix}/rec_loss']
        self.log(f'val{suffix}/rec_loss', rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'val{suffix}/aeloss', aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f'val{suffix}/rec_loss']
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        print('lr_d', lr_d)
        print('lr_g', lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.quantize.parameters()) + list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters()), lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print('Setting up LambdaLR scheduler...')
            scheduler = [{'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}, {'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
            return ([opt_ae, opt_disc], scheduler)
        return ([opt_ae, opt_disc], [])

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log['inputs'] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log['inputs'] = x
        log['reconstructions'] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3:
                    xrec_ema = self.to_rgb(xrec_ema)
                log['reconstructions_ema'] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer('colorize', torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

class AutoencoderKL(pl.LightningModule):

    def __init__(self, ddconfig, lossconfig, embed_dim, ckpt_path=None, ignore_keys=[], image_key='image', colorize_nlabels=None, monitor=None):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig['double_z']
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig['z_channels'], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer('colorize', torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location='cpu')['state_dict']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print('Deleting key {} from state_dict.'.format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f'Restored from {path}')

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return (dec, posterior)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split='train')
            self.log('aeloss', aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split='train')
            self.log('discloss', discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split='val')
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split='val')
        self.log('val/rec_loss', log_dict_ae['val/rec_loss'])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return ([opt_ae, opt_disc], [])

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log['samples'] = self.decode(torch.randn_like(posterior.sample()))
            log['reconstructions'] = xrec
        log['inputs'] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer('colorize', torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

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

class DDIMSampler(object):

    def __init__(self, model, schedule='linear', **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device('cuda'):
                attr = attr.to(torch.device('cuda'))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize='uniform', ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)))
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None, quantize_x0=False, eta=0.0, mask=None, x0=None, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1.0, unconditional_conditioning=None, **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f'Warning: Got {cbs} conditionings but batch-size is {batch_size}')
            elif conditioning.shape[0] != batch_size:
                print(f'Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}')
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        samples, intermediates = self.ddim_sampling(conditioning, size, callback=callback, img_callback=img_callback, quantize_denoised=quantize_x0, mask=mask, x0=x0, ddim_use_original_steps=False, noise_dropout=noise_dropout, temperature=temperature, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, x_T=x_T, log_every_t=log_every_t, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
        return (samples, intermediates)

    @torch.no_grad()
    def ddim_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=False, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, img_callback=None, log_every_t=100, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and (not ddim_use_original_steps):
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f'Running DDIM Sampling with {total_steps} timesteps')
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps, quantize_denoised=quantize_denoised, temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        return (img, intermediates)

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        b, *_, device = (*x.shape, x.device)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        if score_corrector is not None:
            assert self.model.parameterization == 'eps'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return (x_prev, pred_x0)

class DDPM(pl.LightningModule):

    def __init__(self, unet_config, timesteps=1000, beta_schedule='linear', loss_type='l2', ckpt_path=None, ignore_keys=[], load_only_unet=False, monitor='val/loss', use_ema=True, first_stage_key='image', image_size=256, channels=3, log_every_t=100, clip_denoised=True, linear_start=0.0001, linear_end=0.02, cosine_s=0.008, given_betas=None, original_elbo_weight=0.0, v_posterior=0.0, l_simple_weight=1.0, conditioning_key=None, parameterization='eps', scheduler_config=None, use_positional_encodings=False, learn_logvar=False, logvar_init=0.0):
        super().__init__()
        assert parameterization in ['eps', 'x0'], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f'{self.__class__.__name__}: Running in {self.parameterization}-prediction mode')
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f'Keeping EMAs of {len(list(self.model_ema.buffers()))}.')
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule='linear', timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))
        if self.parameterization == 'eps':
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == 'x0':
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError('mu not supported')
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')

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

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return (mean, variance, log_variance)

    def predict_start_from_noise(self, x_t, t, noise):
        return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return (model_mean, posterior_variance, posterior_log_variance)

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = (*x.shape, x.device)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return (img, intermediates)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)
        loss_dict = {}
        if self.parameterization == 'eps':
            target = noise
        elif self.parameterization == 'x0':
            target = x_start
        else:
            raise NotImplementedError(f'Paramterization {self.parameterization} not yet supported')
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss': loss})
        return (loss, loss_dict)

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return (loss, loss_dict)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('global_step', self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log['inputs'] = x
        diffusion_row = list()
        x_start = x[:n_row]
        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)
        log['diffusion_row'] = self._get_rows_from_list(diffusion_row)
        if sample:
            with self.ema_scope('Plotting'):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
            log['samples'] = samples
            log['denoise_row'] = self._get_rows_from_list(denoise_row)
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

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

class DiffusionWrapper(pl.LightningModule):

    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list=None, c_crossattn: list=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()
        return out

class PLMSSampler(object):

    def __init__(self, model, schedule='linear', **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device('cuda'):
                attr = attr.to(torch.device('cuda'))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize='uniform', ddim_eta=0.0, verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)
        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)))
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None, quantize_x0=False, eta=0.0, mask=None, x0=None, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1.0, unconditional_conditioning=None, **kwargs):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f'Warning: Got {cbs} conditionings but batch-size is {batch_size}')
            elif conditioning.shape[0] != batch_size:
                print(f'Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}')
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')
        samples, intermediates = self.plms_sampling(conditioning, size, callback=callback, img_callback=img_callback, quantize_denoised=quantize_x0, mask=mask, x0=x0, ddim_use_original_steps=False, noise_dropout=noise_dropout, temperature=temperature, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, x_T=x_T, log_every_t=log_every_t, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
        return (samples, intermediates)

    @torch.no_grad()
    def plms_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=False, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, img_callback=None, log_every_t=100, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and (not ddim_use_original_steps):
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0, timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f'Running PLMS Sampling with {total_steps} timesteps')
        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img
            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps, quantize_denoised=quantize_denoised, temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        return (img, intermediates)

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1.0, unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = (*x.shape, x.device)

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            if score_corrector is not None:
                assert self.model.parameterization == 'eps'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            return e_t
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return (x_prev, pred_x0)
        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)
        return (x_prev, pred_x0, e_t)

class AssetSelectAgent:

    def __init__(self, config):
        self.config = config
        self.asset_bank = {'audi': 'Audi_Q3_2023.blend', 'benz_g': 'Benz_G.blend', 'benz_s': 'Benz_S.blend', 'mini': 'BMW_mini.blend', 'cadillac': 'Cadillac_CT6.blend', 'chevrolet': 'Chevrolet.blend', 'dodge': 'Dodge_SRT_Hellcat.blend', 'ferriari': 'Ferriari_f150.blend', 'lamborghini': 'Lamborghini.blend', 'rover': 'Land_Rover_range_rover.blend', 'tank': 'M1A2_tank.blend', 'police_car': 'Police_car.blend', 'porsche': 'Porsche-911-4s-final.blend', 'tesla_cybertruck': 'Tesla_cybertruck.blend', 'tesla_roadster': 'Tesla_roadster.blend', 'loader_truck': 'obstacles/Loader_truck.blend', 'bulldozer': 'obstacles/Bulldozer.blend', 'cement': 'obstacles/Cement_isolation_pier.blend', 'excavator': 'obstacles/Excavator.blend', 'sign_fence': 'obstacles/Sign_fence.blend', 'cone': 'obstacles/Traffic_cone.blend'}
        self.assets_dir = config['assets_dir']

    def llm_selecting_asset(self, scene, message):
        try:
            q0 = "I will provide you with an operation statement to add and place a vehicle, and I need you to determine the car's color and type. "
            q1 = 'You need to return a JSON dictionary with 2 keys, including '
            q2 = "(1) 'color', representing in RGB with range from 0 to 255. If the color is not mentioned, the value is just 'default'."
            q3 = "(2) 'type', one of [audi, benz_g, benz_s, mini, cadillac, chevrolet, dodge, ferriari, lamborghini, rover, tank, police_car, porsche, tesla_cybertruck, tesla_roadster, cone, loader_truck, bulldozer, cement, excavator, sign_fence, random]. If the type is not mentioned or not in the type list, it defaults to random."
            q4 = "An example: Given operation statement 'add a black Rover at the front', you should return: {'color':[0,0,0], 'type':'Rover'}"
            q5 = 'Note that you should not return any code or explanations, only provide a JSON dictionary.'
            q6 = 'The operation statement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': "You are an assistant helping me to determine a car's color and type."}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Asset Agent LLM] deciding asset type and color', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            color_and_type = eval(answer)
            color_and_type['type'] = color_and_type['type'] if color_and_type['type'] != 'random' else random.choice(list(self.asset_bank.keys()))
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {color_and_type} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Asset Agent LLM] deciding asset type and color fails.'
        return color_and_type

    def llm_revise_added_cars(self, scene, message, added_car_dict):
        """ This function is a little go beyond asset_select_agent's role. It also consider the motion of the car

        It determine how to modify the dictionary about already added cars
        """
        try:
            q0 = 'I will provide you with a dictionary in which each key is a vehicle id, and each value is the status description of the vehicle in the scene.'
            q1 = 'Specifically, description of the vehicle is also a dictionary. It has keys as follows:'
            q2 = "(1) 'x', vehicle's x position in meter. positive x is heading forward (2) 'y', vehicle's y position in meter. positive y is heading left " + "(3) 'color', vehicle's color in RGB. 'color' would be 'default' or a list represent the RGB values. If the color is not mentioned, the value is just 'default'."
            q3 = "(4) 'type', one of [audi, benz_g, benz_s, mini, cadillac, chevrolet, dodge, ferriari, lamborghini, rover, tank, police_car, porsche, tesla_cybertruck, tesla_roadster, cone, loader_truck, bulldozer, cement, excavator, sign_fence]. "
            q4 = "(5) 'action', vehicle's driving action, one of ['random', 'straight', 'turn left', 'turn right', 'change lane left', 'change lane right', 'static', 'back']"
            q5 = "(6) 'speed', vehicle's driving speed, one of ['random', 'fast', 'slow']"
            q6 = "(7) 'direction', one of ['away', 'close']. In ego view, moving forward is 'away' while moving towards is 'close'."
            q7 = 'I will get you a requirement. To follow my requirement, you should first find out which car I am describing, and then modify its status description dictionary according to my requirement.                 For unmentioned properties, keep them unchanged.'
            q8 = f'Now the dictionary is {added_car_dict}, and my requirement is {message}. '
            q9 = "Note that you should return a JSON dictionary, which only containing the specfic car in requirement with its modified status.                 Just return the JSON dictionary, I'm not asking you to write code."
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me modify and return dictionaries.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Asset Select Agent LLM] revising added cars', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            modified_car_dict = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {modified_car_dict} (number={len(modified_car_dict)})\n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Asset Select Agent LLM] revising added cars fails.'
        return modified_car_dict

    def func_retrieve_blender_file(self, scene):
        """Retrieve the path of the asset file given the asset type.
        """
        for car_name, car_info in scene.added_cars_dict.items():
            car_blender_file = self.asset_bank[car_info['type'].casefold()]
            car_info['blender_file'] = os.path.join(self.assets_dir, car_blender_file)

def scale_dense_depth_map(dense_depth_map, sparse_depth_map, depth_map_mask):
    """
    Scale the dense depth map to match the scale of the sparse depth map.

    :param dense_depth_map: [H, W] dense depth map
    :param sparse_depth_map: [H, W] sparse depth map
    :param depth_map_mask: [H, W] mask indicating valid values in the sparse depth map
    :return: Scaled dense depth map
    """
    dense_depth_map = torch.tensor(dense_depth_map, dtype=torch.float32)
    sparse_depth_map = torch.tensor(sparse_depth_map, dtype=torch.float32)
    depth_map_mask = torch.tensor(depth_map_mask, dtype=torch.float32)
    valid_dense_depths = torch.masked_select(dense_depth_map, depth_map_mask.bool())
    valid_sparse_depths = torch.masked_select(sparse_depth_map, depth_map_mask.bool())
    alpha_numerator = torch.sum(valid_dense_depths * valid_sparse_depths)
    alpha_denominator = torch.sum(valid_dense_depths ** 2)
    alpha = alpha_numerator / alpha_denominator
    print('scaling factor: ', alpha)
    scaled_dense_depth_map = alpha * dense_depth_map
    return scaled_dense_depth_map

class DeletionAgent:

    def __init__(self, config):
        self.config = config
        self.inpaint_dir = config['inpaint_dir']
        self.video_inpaint_dir = config['video_inpaint_dir']

    def llm_finding_deletion(self, scene, message, scene_object_description):
        try:
            q0 = 'I will provide you with an operation statement and a dictionary containing information about cars in a scene. ' + ' You need to determine which car or cars should be deleted from the dictionary. '
            q1 = 'The dictionary is ' + str(scene_object_description)
            q2 = 'The keys of the dictionary are the car IDs, and the value is also a dictionary containing car detail, ' + 'including its image coordinate (u,v) in an image frame, depth, color in RGB.'
            q2 = "My statement may include information about the car's color or position. You should find out from my statement which cars should be deleted and return their car IDs"
            q3 = 'Note: (1) The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. ' + "And the larger the 'u', the more to the right; And the larger the 'v', the more to the down. " + "(2) You can judge the distance by the 'depth'. The greater the depth, the farther the distance, the smaller the depth, the closer the distance." + '(3) The description of the color may not be absolutely accurate, choose the car with the closest color.'
            q4 = "You should return a JSON dictionary, with a key: 'removed_cars'." + " 'removed_cars' contains IDs of all the cars that meet the requirements. "
            q5 = 'Note that there is no need to return any code or explanations; only provide a JSON dictionary.'
            q6 = "If the dictionary is empty, 'removed_cars' should be an empty list "
            q7 = 'The requirement is :' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to assess and maintain information in a dictionary.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Deletion Agent LLM] finding the car to delete', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            deletion_car_ids = eval(answer)['removed_cars']
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {deletion_car_ids} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('[Deletion Agent LLM] finding the car to delete fails')
            return []
        return deletion_car_ids

    def llm_putting_back_deletion(self, scene, message, scene_object_description):
        try:
            deleted_object_dict = {k: v for k, v in scene_object_description.items() if k in scene.removed_cars}
            q0 = 'I will provide you with a dictionary in which each key is a vehicle id, and each value is the description of the vehicle in the image.'
            q1 = "Specifically, description of the vehicle is also a dictionary. It has keys: (1) vehicle's u in image coordinate (2) vehicle's v in image coordinate (3) vehicle color in RGB. (4) vehicle's depth from viewpoint"
            q2 = 'The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. ' + "The larger the 'u', the more to the right; And the larger the 'v', the more to the down. "
            q3 = 'I will get you a requirement, and I want you can follow this requirement and take out all the relavant vehicle ids from the dictionary.'
            q4 = f'Now the dictionary is {deleted_object_dict}, and my requirement is {message}. My requirement may contain extraneous verb descriptions or the wrong singular and plural expression, please ignore.'
            q5 = "Note that you should return a JSON dictionary, the key is 'selected_vehicle', the value includes the vehicle ids. DO NOT return anything else. I'm not asking you to write code."
            prompt_list = [q0, q1, q2, q3, q4, q5]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me maintain and return dictionaries.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Deletion Agent LLM] finding the car to be put back', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            put_back_car_ids = eval(answer)['selected_vehicle']
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {put_back_car_ids} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('[Deletion Agent LLM] finding the car to be put back fails')
        return put_back_car_ids

    def func_inpaint_scene(self, scene):
        """
        Call inpainting, store results in scene.current_inpainted_images

        if no scene.removed_cars
            just return

        """
        if len(scene.removed_cars) == 0:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} No inpainting.')
            scene.current_inpainted_images = scene.current_images
            return
        current_dir = os.getcwd()
        inpaint_input_path = os.path.join(current_dir, scene.cache_dir, 'inpaint_input')
        inpaint_output_path = os.path.join(current_dir, scene.cache_dir, 'inpaint_output')
        check_and_mkdirs(inpaint_input_path)
        check_and_mkdirs(inpaint_output_path)
        if scene.is_ego_motion is False:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is False, inpainting one frame.')
            all_mask = self.func_get_mask(scene)
            img = scene.current_images[0]
            masked_img = copy.deepcopy(img)
            if scene.is_wide_angle:
                masked_img = cv2.resize(masked_img, (1152, 256))
            else:
                masked_img = cv2.resize(masked_img, (512, 384))
            imageio.imwrite(os.path.join(inpaint_input_path, 'img.png'), masked_img.astype(np.uint8))
            imageio.imwrite(os.path.join(inpaint_input_path, 'img_mask.png'), all_mask.astype(np.uint8))
            current_dir = os.getcwd()
            os.chdir(self.inpaint_dir)
            os.system(f'python scripts/inpaint.py --indir {inpaint_input_path} --outdir {inpaint_output_path}')
            os.chdir(current_dir)
            new_img = imageio.imread(os.path.join(inpaint_output_path, 'img.png'))
            new_img = cv2.resize(new_img, (scene.width, scene.height))
            all_mask_in_ori_resolution = cv2.resize(all_mask, (scene.width, scene.height)).reshape(scene.height, scene.width, 1).repeat(3, axis=2)
            new_img = np.where(all_mask_in_ori_resolution == 0, scene.current_images[0], new_img)
            scene.current_inpainted_images = [new_img] * scene.frames
        else:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is True, inpainting multiple frame (as video).')
            mask_list = []
            for i in range(scene.frames):
                current_frame_mask = np.zeros((scene.height, scene.width))
                for car_id in scene.bbox_data.keys():
                    if scene.bbox_car_id_to_name[car_id] in scene.removed_cars:
                        corners = generate_vertices(scene.bbox_data[car_id])
                        mask, mask_corners = get_outlines(corners, transform_nerf2opencv_convention(scene.current_extrinsics[i]), scene.intrinsics, scene.height, scene.width)
                        current_frame_mask[mask == 1] = 1
                mask_list.append(current_frame_mask)
            np.save(f'{self.video_inpaint_dir}/chatsim/masks.npy', mask_list)
            np.save(f'{self.video_inpaint_dir}/chatsim/current_images.npy', scene.current_images)
            current_dir = os.getcwd()
            os.chdir(self.video_inpaint_dir)
            os.system(f'python remove_anything_video_npy.py                         --dilate_kernel_size 15                         --lama_config lama/configs/prediction/default.yaml                         --lama_ckpt ./pretrained_models/big-lama                         --tracker_ckpt vitb_384_mae_ce_32x4_ep300                         --vi_ckpt ./pretrained_models/sttn.pth                         --mask_idx 2                         --fps 25')
            os.chdir(current_dir)
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} Video Inpainting Done!')
            inpainted_images = np.load(f'{self.video_inpaint_dir}/chatsim/inpainted_imgs.npy', allow_pickle=True)
            scene.current_inpainted_images = [np.array(image) for image in inpainted_images]

    def func_get_mask(self, scene):
        masks = []
        extrinsic_for_project = transform_nerf2opencv_convention(scene.current_extrinsics[0])
        for car_name in scene.removed_cars:
            car_id = scene.name_to_bbox_car_id[car_name]
            corners = generate_vertices(scene.bbox_data[car_id])
            mask, _ = get_outlines(corners, extrinsic_for_project, scene.intrinsics, scene.height, scene.width)
            mask *= 255
            masks.append(mask)
        mask = np.max(np.stack(masks), axis=0)
        if scene.is_wide_angle:
            mask = cv2.resize(mask, (1152, 256))
        else:
            mask = cv2.resize(mask, (512, 384))
        return mask

class MotionAgent:

    def __init__(self, config):
        self.config = config
        self.motion_tracking = config.get('motion_tracking', False)

    def llm_reasoning_dependency(self, scene, message):
        """ LLM reasoning of Motion Agent, determine if the vehicle placement is depend on scene elements.
        Input:
            scene : Scene
                scene object.
            message : str
                language prompt to ChatSim.
        """
        try:
            q0 = 'I will provide an operation statement to add a vehicle, and you need to determine whether the position of the added car has any spatial dependency with other cars in my statement'
            q1 = "Only return a JSON format dictionary as your response, which contains a key 'dependency'. If the added car's position depends on other objects, set it to 1; otherwise, set it to 0."
            q2 = "An Example: Given statement 'add an Audi in the back which drives ahead', you should return {'dependency': 0}. This is because I only mention the added Audi."
            q3 = "An Example: Given statement 'add a Porsche at 2m to the right of the red Audi.', you should return {'dependency': 1}. This is because Porsche's position depends on Audi."
            q4 = "An Example: Given statement 'add a car in front of me.', you should return {'dependency': 0}. This is because 'me' is not other car in the scene."
            q5 = 'The statement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to extract information from the operations.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] analyzing insertion scene dependency ', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_mode = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_mode} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] reasoning object dependency fails.'
        return placement_mode

    def llm_placement_wo_dependency(self, scene, message):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, and I need you to extract 3 specific placement information from the statement, including: '
            q1 = " (1) 'mode', one of ['front', 'left front', 'left', 'right front', 'right', 'random'], representing approximate initial positions of the vehicle. If not specified, it defaults to 'random'."
            q2 = " (2) 'distance_constraint' indicates whether there's a constraint on the distance of the added vehicle. 0 means no constraint, 1 means there is a constraint." + " If there's no relevant information mentioned, it defaults to 0."
            q3 = " (3) 'distance_min_max' represents the range of constraints when 'distance_constraint' applicable. It should be a tuple in the format (min, max), for example, (9, 11) means the minimum distance is 9, and the maximum is 11." + " When there's 'distance_constraint' is 0, the default value is (4, 45). If distance is specified as a specific value 'x', 'distance_min_max' is (x, x+5)"
            q4 = "Just return the json dict with keys:'mode', 'distance_constraint', 'distance_min_max'. Do not return any code or discription."
            q5 = "An Example: Given operation statement: 'Add an Audi 7-10 meters ahead', you should return " + "{'mode':'front', 'distance_constraint': 1, 'distance_min_max':(7,10)}"
            q6 = "An Example: Given operation statement: 'Add an Porsche in the right front.', you should return " + "{'mode':'right front', 'distance_constraint': 0, 'distance_min_max':(4, 45)}"
            q7 = 'Note that you should not return any code or explanations, only provide a JSON dictionary.'
            q8 = 'The operation statement:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to determine how to place a car.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] deciding scene-independent object placement', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_prior = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] deciding placement fails.'
        return placement_prior

    def llm_placement_w_dependency(self, scene, message, scene_object_description):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, as well as information of other cars in the scene.'
            q1 = 'I need you to determine a specific position (x, y) for placement of the added car in my statement. '
            q2 = 'Information of other cars in the scene is a two-level dictionary, with the first level representing the different car id in the scene, ' + 'and the second level containing various information about that car, including the (x, y) of its world 3D coordinate, ' + 'its image coordinate (u, v) in an image frame, depth, and rgb color representation.'
            q3 = 'The dictionary is' + str(scene_object_description)
            q4 = 'I will also further inform you about the operations that have been previously performed on this scene. ' + 'You can use these past operations, along with the dictionary I provide, to generate the final position.'
            q5 = 'The previously performed operation is : ' + str(scene.past_operations)
            q6 = "If the car with key 'direction', and direction is close, 'behind' means keep the same 'y' and increase 'x' 10 meters. If direction is away, 'behind' means keep the same 'y' and decrease 'x' 10 meters." + "If the car with key 'direction', and direction is close, 'front' means keep the same 'y' and decrease 'x' 10 meters. If direction is away, 'front' means keep the same 'y' and increase 'x' 10 meters."
            q7 = "'left' means keep the same 'x' and increase 'y' 5m, 'right' means keep the same 'x' and decrease 'y' 5m."
            q8 = "You should return a placemenet positon in JSON dictionary with 2 keys: 'x', 'y'. Do not provide any code or explanations, only return the final JSON dictionary."
            q9 = 'The requirement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to determine how to place a car.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] deciding scene-dependent object placement', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            placement_prior = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {placement_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] deciding placement fails.'
        return placement_prior

    def llm_motion_planning(self, scene, message):
        try:
            q0 = 'I will provide you with an operation statement to add and place a vehicle, and I need you to determine the its motion situation from my statement, including: '
            q1 = "(1) 'action', one of ['static', 'random', 'straight', 'turn left', 'turn right']. If action not mentioned in the statement, it defaults to 'straight'." + "For example, the statement is 'add a black car in front of me', then the action is 'straight'."
            q2 = "(2) 'speed', the approximate speed of the vehicle, one of ['random', 'fast', 'slow']. If speed is not mentioned in the statement, it defaults to 'slow'."
            q3 = "(3) 'direction', one of ['away', 'close', 'random']. 'away' represents the direction away from oneself, and 'close' represents the direction toward oneself." + "For example, moving forward is 'away' from oneself, while moving towards oneself is 'close'. If direction is not mentioned in the statement, just return 'random'."
            q4 = "(4) 'wrong_way', if the vehicle drives in the wrong way, one of ['true'. 'false']. If the information is not mentioned in the statement, it defaults to 'false'."
            q4 = "An Example: Given the statement 'add a Tesla that is racing straight ahead in the right front of the scene', you should return {'action': 'straight', 'speed': 'fast', 'direction': 'away', 'wrong_way': 'false'}"
            q5 = "An Example: Given the statement 'add a yellow Audi in front of the scene', you should return {'action': 'static', 'speed': 'random', 'direction': 'away', 'wrong_way': 'false'}"
            q6 = "An Example: Given the statement 'add a Tesla coming from the front and driving in the wrong way', you should return {'action': 'straight', 'speed': 'random', 'direction': 'close', 'wrong_way': 'true'}"
            q7 = 'Note that there is no need to return any code or explanations; only provide a JSON dictionary. Do not include any additional statements.'
            q8 = 'The operation statement is:' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to assess the motion situation for adding vehicles.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Motion Agent LLM] finding motion prior', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            motion_prior = eval(answer)
            if not motion_prior.get('wrong_way'):
                motion_prior['wrong_way'] = False
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {motion_prior} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[Motion Agent LLM] finding motion prior fails.'
        return motion_prior

    def func_placement_and_motion_single_vehicle(self, scene, added_car_name):
        added_car_id = added_car_name.lstrip('added_car_')
        transformed_map_data_ = transform_node_to_lane(scene.map_data)
        all_current_vertices_coord = scene.all_current_vertices_coord
        for added_traj in scene.all_trajectories:
            all_current_vertices_coord = np.vstack([all_current_vertices_coord, added_traj[0:1, 0:2]])
        one_added_car = scene.added_cars_dict[added_car_name]
        if one_added_car['need_placement_and_motion'] is True:
            scene.added_cars_dict[added_car_name]['need_placement_and_motion'] = False
            one_added_car = scene.added_cars_dict[added_car_name]
            transformed_map_data = deepcopy(transformed_map_data_)
            if one_added_car['wrong_way'] is True:
                transformed_map_data['centerline'][:, -1] = (transformed_map_data['centerline'][:, -1] + 1) % 2
                transformed_map_data['centerline'] = np.concatenate((transformed_map_data['centerline'][:, 2:4], transformed_map_data['centerline'][:, 0:2], transformed_map_data['centerline'][:, 4:]), axis=1)
                transformed_map_data['centerline'] = np.flip(transformed_map_data['centerline'], axis=0)
            if one_added_car.get('x') is None:
                placement_result = vehicle_placement(transformed_map_data, all_current_vertices_coord, one_added_car['direction'] if one_added_car['direction'] != 'random' else random.choice(['away', 'close']), one_added_car['mode'], one_added_car['distance_constraint'], one_added_car['distance_min_max'], 'default')
            else:
                placement_result = vehicle_placement_specific(transformed_map_data, all_current_vertices_coord, np.array([one_added_car['x'], one_added_car['y']]))
            if placement_result[0] is None:
                del scene.added_cars_dict[added_car_name]
                return
            one_added_car['placement_result'] = placement_result
            try:
                motion_result = vehicle_motion(transformed_map_data, scene.all_current_vertices[:, ::2, :2] if scene.all_current_vertices.shape[0] != 0 else scene.all_current_vertices, placement_result=one_added_car['placement_result'], high_level_action_direction=one_added_car['action'], high_level_action_speed=one_added_car['speed'], dt=1 / scene.fps, total_len=scene.frames)
            except ValueError as e:
                print(f'{colored('[Motion Agent] Error: Potentially no feasible destination can be found.', color='red', attrs=['bold'])} {e}')
                raise ValueError('No feasible destination can be found.')
            if motion_result[0] is None:
                del scene.added_cars_dict[added_car_name]
                return
            one_added_car['motion'] = motion_result
            scene.added_cars_dict[added_car_name] = one_added_car
            all_trajectories = []
            for one_car_name in scene.added_cars_dict.keys():
                all_trajectories.append(scene.added_cars_dict[one_car_name]['motion'][:, :2])
            all_trajectories_after_check_collision = check_collision_and_revise_dynamic(all_trajectories)
            all_trajectories_after_check_collision = all_trajectories
            scene.all_trajectories = all_trajectories_after_check_collision
            for idx, one_car_name in enumerate(scene.added_cars_dict.keys()):
                motion_result = all_trajectories_after_check_collision[idx]
                placement_result = scene.added_cars_dict[one_car_name]['placement_result']
                direction = np.zeros((motion_result.shape[0], 1))
                angle = np.arctan2(placement_result[-1] - placement_result[-3], placement_result[-2] - placement_result[-4])
                for i in range(motion_result.shape[0] - 1):
                    if motion_result[i, 0] == motion_result[i + 1, 0] and motion_result[i, 1] == motion_result[i + 1, 1]:
                        direction[i, 0] = angle
                    else:
                        direction[i, 0] = np.arctan2(motion_result[i + 1, 1] - motion_result[i, 1], motion_result[i + 1, 0] - motion_result[i, 0])
                direction[-1, 0] = direction[-2, 0]
                motion_result = np.concatenate((motion_result, direction), axis=1)
                if self.motion_tracking:
                    try:
                        from simulator import TrajectoryTracker
                    except ModuleNotFoundError:
                        error_msg1 = f'{colored('[ERROR]', color='red', attrs=['bold'])} Trajectory Tracking Module is not installed.\n'
                        error_msg2 = "\nYou can 1) Install Installation README's Step 5: Setup Trajectory Tracking Module"
                        error_msg3 = "\n     Or 2) set ['motion_agent']['motion_tracking'] to False in config.\n"
                        raise ModuleNotFoundError(error_msg1 + error_msg2 + error_msg3)
                    reference_line = interpolate_uniformly(motion_result, int(scene.frames * scene.fps / 10))
                    reference_line = [(reference_line[i, 0], reference_line[i, 1]) for i in range(reference_line.shape[0])]
                    init_state = (motion_result[0, 0], motion_result[0, 1], motion_result[0, 2], np.linalg.norm(np.array(reference_line[1]) - np.array(reference_line[0])) * 10)
                    pretrained_checkpoint_dir = './chatsim/foreground/drl-based-trajectory-tracking/submodules/drltt-assets/checkpoints/track/checkpoint'
                    trajectory_tracker = TrajectoryTracker(checkpoint_dir=pretrained_checkpoint_dir)
                    states, actions = trajectory_tracker.track_reference_line(reference_line=reference_line, init_state=init_state)
                    motion_result = np.stack(states)[:, :-1]
                    motion_result = interpolate_uniformly(motion_result, scene.frames)
                scene.added_cars_dict[one_car_name]['motion'] = motion_result

class ViewAdjustAgent:

    def __init__(self, config):
        self.config = config

    def llm_reasoning_ego_motion(self, scene, message):
        try:
            q0 = 'I will give you a description about view adjustment, I need you to help me judge if the description is related to static view adjust or ego is dynamic(with motion).'
            q1 = "Given my description, return a dictionary in JSON format, with key 'if_view_motion'"
            q2 = "If the description is just a view adjust operation, the 'if_view_motion' should be 0. If the description is related to view motion, the 'if_view_motion' should be 1."
            q3 = "I will give you some examples. <user>: Rotate the viewpoint 30 degrees to the left, you should return {'if_view_motion':0}. " + "<user>: viewpoint moves ahead slowly, you should return {'if_view_motion':1}. "
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to provide information and ultimately return a JSON dictionary.'}, {'role': 'user', 'content': q0}, {'role': 'user', 'content': q1}, {'role': 'user', 'content': q2}, {'role': 'user', 'content': q3}, {'role': 'user', 'content': message}])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[View Adjust Agent LLM] reasoning the view motion', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            if_view_motion = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {if_view_motion} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[View Adjust Agent LLM] fails, can not recongnize instruction'
        if if_view_motion['if_view_motion'] == 0:
            return False
        else:
            return True

    def llm_view_motion_gen(self, scene, message):
        try:
            q0 = 'I will give you a description about ego motion, you should tell me the speed of ego.'
            q1 = "Given my description, return a dictionary in JSON format, with key 'speed'."
            q2 = "If the ego motion is fast, 'speed' should be 'fast'; if the ego motion is slow, 'speed' should be 'slow'; if the description doesnot mention speed, 'speed' is default as 'fast'."
            q3 = "I will give you some examples. <user>: ego vehicle moves forward, you should return {'speed':'fast'}. " + "<user>: ego vehicle drives ahead slowly, you should return {'speed':'slow'}. "
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to provide information and ultimately return a JSON dictionary.'}, {'role': 'user', 'content': q0}, {'role': 'user', 'content': q1}, {'role': 'user', 'content': q2}, {'role': 'user', 'content': q3}, {'role': 'user', 'content': message}])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[View Adjust Agent LLM] generating the ego motion', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            ego_motion_speed = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {ego_motion_speed} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[View Adjust Agent LLM] fails, can not recongnize instruction'
        if ego_motion_speed['speed'] == 'fast':
            return (0, scene.nerf_motion_extrinsics.shape[0])
        else:
            return (0, scene.nerf_motion_extrinsics.shape[0] // 3)

    def llm_view_adjust(self, scene, message):
        try:
            q0 = "I will give you a transformation operation for my viewpoint, which may include translation in 'x', 'y', 'z' or a rotation 'theta' around z-axis. "
            q1 = "For translation, positive 'x' represents forward, positve 'y' represents left, and 'z' represents up. It follows a left-hand coordinate system." + "For rotation, postive 'theta' is counterclockwise. So from own perspective, my viewpoint turns to the left. 'theta' is in degree."
            q2 = "Given my operation, return a dictionary in JSON format, with keys 'x', 'y', 'z', 'theta'."
            q3 = 'I will give you some examples: <user>: Rotate the viewpoint 30 degrees to the left ' + "<assistant>: {\n  'x': 0,\n  'y': 0,\n  'z': 0,\n  'theta': 30,\n } \n" + '<user>: move the viewpoint forward by 1 ' + "<assistant>: {\n  'x': 1,\n  'y': 0,\n  'z': 0,\n  'theta': 0,\n }  \n" + '<user>: move the viewpoint to the right by 1' + "<assistant>: {\n  'x': 0,\n  'y': -1,\n  'z': 0,\n  'theta': 0,\n} "
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to provide information and ultimately return a JSON dictionary.'}, {'role': 'user', 'content': q0}, {'role': 'user', 'content': q1}, {'role': 'user', 'content': q2}, {'role': 'user', 'content': q3}, {'role': 'user', 'content': message}])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[View Adjust Agent LLM] analyzing view change', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            delta_extrinsic = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {delta_extrinsic} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return '[View Adjust Agent LLM] fails, can not recongnize instruction'
        return delta_extrinsic

    def func_update_extrinsic(self, scene, delta_extrinsic):
        scene.current_extrinsics[:, 0, 3] += delta_extrinsic['x']
        scene.current_extrinsics[:, 1, 3] += delta_extrinsic['y']
        scene.current_extrinsics[:, 2, 3] += delta_extrinsic['z']
        theta = delta_extrinsic['theta']
        theta = theta / 180 * np.pi
        T_theta = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        scene.current_extrinsics = np.matmul(T_theta, scene.current_extrinsics)

    def func_generate_extrinsic(self, scene, start_frame_idx, end_frame_idx):
        scene.current_extrinsics = inter_poses(scene.nerf_motion_extrinsics[start_frame_idx:end_frame_idx:3], scene.frames)

class BackgroundRendering3DGSAgent:

    def __init__(self, config):
        self.config = config
        self.is_wide_angle = config['nerf_config']['is_wide_angle']
        self.gs_dir = config['gs_config']['gs_dir']
        self.model_folder = os.path.join(config['gs_config']['gs_dir'], config['gs_config']['output_folder'], config['gs_config']['gs_model_name'])
        self.gs_novel_view_dir = os.path.join(self.model_folder, 'chatsim_novel_views')

    def func_render_background(self, scene):
        """
        Call the NeRF, store results in scene.current_images
        """
        scene.is_ego_motion = not np.all(scene.current_extrinsics == scene.current_extrinsics[0])
        if scene.is_ego_motion:
            print(f'{colored('[Background Gaussian Splatting]', 'red', attrs=['bold'])} is_ego_motion is True, rendering multiple frames')
            camera_extrinsics = scene.current_extrinsics[:, :3, :]
            camera_intrinsics = scene.intrinsics
        else:
            print(f'{colored('[Background Gaussian Splatting]', 'red', attrs=['bold'])} is_ego_motion is False, rendering one frame')
            camera_extrinsics = scene.current_extrinsics[0:1, :3, :]
            camera_intrinsics = scene.intrinsics
        np.savez(os.path.join(self.model_folder, 'chatsim_extint.npz'), camera_extrinsics=camera_extrinsics, camera_intrinsics=camera_intrinsics, H=scene.height, W=scene.width)
        if os.path.exists(self.gs_novel_view_dir) and len(os.listdir(self.gs_novel_view_dir)) > 0:
            os.system(f'rm -r {self.gs_novel_view_dir}/*')
        current_dir = os.getcwd()
        os.chdir(self.gs_dir)
        render_command = f'python render_chatsim.py                             --model_path {self.model_folder}'
        os.system(render_command)
        os.chdir(current_dir)
        scene.current_images = []
        img_rendered_pkls = os.listdir(self.gs_novel_view_dir)
        assert len(img_rendered_pkls) == 1, f'the folder has {len(img_rendered_pkls)} files'
        img_rendered_pkl = os.path.join(self.gs_novel_view_dir, img_rendered_pkls[0])
        with open(img_rendered_pkl, 'rb') as f:
            scene.current_images = pickle.load(f)
        if not scene.is_ego_motion:
            scene.current_images = scene.current_images * scene.frames

class BackgroundRenderingAgent:

    def __init__(self, config):
        self.config = config
        self.is_wide_angle = config['nerf_config']['is_wide_angle']
        self.scene_name = config['nerf_config']['scene_name']
        self.f2nerf_dir = config['nerf_config']['f2nerf_dir']
        self.nerf_exp_name = config['nerf_config']['nerf_exp_name']
        self.f2nerf_config = config['nerf_config']['f2nerf_config']
        self.dataset_name = config['nerf_config']['dataset_name']
        self.nerf_mode = config['nerf_config']['rendering_mode']
        self.nerf_exp_dir = os.path.join(self.f2nerf_dir, 'exp', self.scene_name, self.nerf_exp_name)
        self.nerf_data_dir = os.path.join(self.f2nerf_dir, 'data', self.dataset_name, self.scene_name)
        nerf_output_foler_name = 'wide_angle_novel_images' if self.is_wide_angle else 'novel_images'
        self.nerf_novel_view_dir = os.path.join(self.nerf_exp_dir, nerf_output_foler_name)
        self.nerf_quiet_render = config['nerf_config']['nerf_quiet_render']
        if self.is_wide_angle:
            assert 'wide' in self.nerf_mode
        else:
            assert 'wide' not in self.nerf_mode

    def func_render_background(self, scene):
        """
        Call the NeRF, store results in scene.current_images
        """
        scene.is_ego_motion = not np.all(scene.current_extrinsics == scene.current_extrinsics[0])
        if scene.is_ego_motion:
            print(f'{colored('[Mc-NeRF]', 'red', attrs=['bold'])} is_ego_motion is True, rendering multiple frames')
            poses_render = scene.current_extrinsics[:, :3, :]
            np.save(os.path.join(self.nerf_data_dir, 'poses_render.npy'), poses_render)
            if os.path.exists(self.nerf_novel_view_dir) and len(os.listdir(self.nerf_novel_view_dir)) > 0:
                os.system(f'rm -r {self.nerf_novel_view_dir}/*')
            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir)
            render_command = f'python scripts/run.py                                 --config-name={self.f2nerf_config}                                 dataset_name={self.dataset_name}                                 case_name={self.scene_name}                                 exp_name={self.nerf_exp_name}                                 mode={self.nerf_mode}                                 is_continue=true                                 +work_dir={os.getcwd()}'
            if self.nerf_quiet_render:
                render_command += ' > /dev/null 2>&1'
            os.system(render_command)
            os.chdir(current_dir)
            scene.current_images = []
            img_path_list = os.listdir(self.nerf_novel_view_dir)
            img_path_list.sort(key=lambda x: int(x[:-4]))
            for img_path in img_path_list:
                scene.current_images.append(imageio.imread(os.path.join(self.nerf_novel_view_dir, img_path))[:, :scene.width])
        else:
            print(f'{colored('[Mc-NeRF]', 'red', attrs=['bold'])} is_ego_motion is False, rendering one frame')
            poses_render = scene.current_extrinsics[0:1, :3, :]
            np.save(os.path.join(self.nerf_data_dir, 'poses_render.npy'), poses_render)
            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir)
            render_command = f'python scripts/run.py                                 --config-name={self.f2nerf_config}                                 dataset_name={self.dataset_name}                                 case_name={self.scene_name}                                 exp_name={self.nerf_exp_name}                                 mode={self.nerf_mode}                                 is_continue=true                                 +work_dir={os.getcwd()}'
            if self.nerf_quiet_render:
                render_command += ' > /dev/null 2>&1'
            os.system(render_command)
            os.chdir(current_dir)
            novel_image = imageio.imread(os.path.join(self.nerf_novel_view_dir, '50000_000.png'))[:, :scene.width]
            scene.current_images = [novel_image] * scene.frames

class ProjectManager:

    def __init__(self, config):
        self.config = config

    def decompose_prompt(self, scene, user_prompt):
        """ decompose the prompt to the corresponding chatsim.agents.
        Input:
            scene : Scene
                scene object.
            user_prompt : str
                language prompt to ChatSim.
        Return:
            tasks : dict
                a dictionary of decomposed tasks.
        """
        q0 = 'I have a requirement of editing operations in an autonomous driving scenario, and I need your help to break it down into one or several supportable actions. The scene is large which means many vehicles can be contained. '
        q1 = 'The supportable five actions include adding vehicles ,                 deleting vehicles ,                 put back deleted vehicles,                 adjusting added vehicles ,                 viewpoint adjustment.'
        q2 = 'Please try to retain all the semantics and adjunct words from the original text. Each adding action should only contain one car. ' + 'Information about adding vehicles (such as their type, positions, driving status, speed, color, etc.) should be directly included within the adding action.'
        q3 = 'Split actions should be stored in a JSON dictonary. The key is action id and the value is specific action. They will be executed sequentially, and the broken operations should be independent with each other and do not rely on the detailed scene information.'
        q4 = "An example: the requirement is 'substitute the red car in the scene', you break it down and return" + "{ 1: 'Delete the red car from the scene', 2: 'Add a new car at the location where the red car was deleted' }."
        q5 = "An example: the requirement is 'delete the farthest car and add a red Audi in the right front', you break it down and return " + "{ 1: 'Delete the farthest car', 2: 'Add a red Audi in the right front' }"
        q6 = "An example: the requirement is 'delete all cars', you break it down and return " + "{ 1: 'Delete all the cars'} "
        q7 = 'I may provide very abstract requirements. For such requirements, you should analyze how to comply with the splitting of actions.'
        q8 = "An example (very abstract): the requirement is 'I want several cars driving slowly in the scene', you analyse and return " + "{ 1: 'Add one car driving slowly', 2 : 'Add one car driving slowly', 3 : 'Add one car driving slowly', 4 : 'Add one car driving slowly', 5 : 'Add one car driving slowly', 6 : 'Add one car driving slowly', 7 : 'Add one car driving slowly'} "
        q9 = 'The scene is large enough to contain more than 20 vehicles. So many vehicles can be added to the scene. Do not return any code or explanation; only a JSON dictionary is required.'
        q10 = 'Attention: the adjustments for one specific added vehicle should be included in one single output action. If there are multiple adjustments for one already added car, these adjustments must be merged in one action.'
        q11 = 'Attention: Do not appear information about the vehicles in the other broken actions.'
        q12 = 'The requirement is:' + user_prompt
        prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
        result = openai.ChatCompletion.create(model='gpt-4-turbo-preview', messages=[{'role': 'system', 'content': 'You are an assistant helping me to break down the operations.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
        answer = result['choices'][0]['message']['content']
        print(f'{colored('[User prompt]', color='magenta', attrs=['bold'])} {user_prompt}\n')
        print(f'{colored('[Project Manager] decomposing tasks', color='magenta', attrs=['bold'])}                \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
        try:
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            tasks = eval(answer)
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {answer} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            return 'Can not parse the requirement.'
        return tasks

    def dispatch_task(self, scene, task, tech_agents):
        """ dispatch the tasks to the corresponding chatsim.agents.
        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        operation_category = {1: 'adding', 2: 'deleting', 3: 'adjusting the viewpoint', 4: 'putting back previously deleted vehicles', 5: 'operating on previously added vehicles'}
        q0 = 'I will provide you with an action, and you will help me determine which operation this action belongs to.'
        q1 = 'Operations include (1) adding (2) deleting, (3) adjusting the viewpoint, (4) putting back previously deleted vehicles, (5) operating on previously added vehicles.'
        q2 = "Return the information in JSON format, with a key named 'operation'."
        q3 = "An Example: Given action 'Remove the red car from the scene', you should return {'operation': 2}"
        q4 = "An Example: Given action 'Add a green Porsche at the location where the red car was removed', you should return {'operation': 1}"
        q5 = "An Example: Given action 'Put back the deleted white car', you should return {'operation': 4}"
        q6 = "An Example: Given action 'Move the car just added to the right by 2m', you should return {'operation': 5}"
        q7 = 'Note that you should not return any code or explanations, only provide a JSON dictionary.'
        q8 = task
        prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7, q8]
        result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to classify operations.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
        answer = result['choices'][0]['message']['content']
        print(f'{colored('[Project Manager] dispatching each task', color='magenta', attrs=['bold'])}                 \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
        start = answer.index('{')
        answer = answer[start:]
        end = answer.rfind('}')
        answer = answer[:end + 1]
        operation = eval(answer)['operation']
        print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {operation}. ({operation_category[operation]}) \n')
        if operation == 1:
            self.addition_operation(scene, task, tech_agents)
        elif operation == 2:
            self.deletion_operation(scene, task, tech_agents)
        elif operation == 3:
            self.view_adjust_operation(scene, task, tech_agents)
        elif operation == 4:
            self.put_back_deleted_operation(scene, task, tech_agents)
        elif operation == 5:
            self.revise_added_operation(scene, task, tech_agents)
        scene.past_operations.append(task)

    def addition_operation(self, scene, task, tech_agents):
        """ addition operation. 
        Participants: asset_select_agent, motion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        asset_select_agent = tech_agents['asset_select_agent']
        motion_agent = tech_agents['motion_agent']
        placement_mode = motion_agent.llm_reasoning_dependency(scene, task)
        if placement_mode['dependency'] == 0:
            placement_prior = motion_agent.llm_placement_wo_dependency(scene, task)
        else:
            valid_object_descriptors_for_cars_in_scene = ['x', 'y', 'u', 'v', 'depth', 'rgb']
            scene_object_description = {}
            for car_name, description_dict in scene.original_cars_dict.items():
                filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors_for_cars_in_scene}
                scene_object_description[car_name] = filtered_description_dict
            valid_object_descriptors_for_added_cars = ['color', 'type']
            for car_name, description_dict in scene.added_cars_dict.items():
                filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors_for_added_cars}
                filtered_description_dict['x'] = description_dict['placement_result'][0]
                filtered_description_dict['y'] = description_dict['placement_result'][1]
                filtered_description_dict['direction'] = description_dict['direction']
                scene_object_description[car_name] = filtered_description_dict
            placement_prior = motion_agent.llm_placement_w_dependency(scene, task, scene_object_description)
        asset_color_and_type = asset_select_agent.llm_selecting_asset(scene, task)
        motion_prior = motion_agent.llm_motion_planning(scene, task)
        added_car_name = scene.add_car({**asset_color_and_type, **placement_prior, **motion_prior})
        motion_agent.func_placement_and_motion_single_vehicle(scene, added_car_name)

    def deletion_operation(self, scene, task, tech_agents):
        """ deletion operation. 
        Participants: deletion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        deletion_agent = tech_agents['deletion_agent']
        valid_object_descriptors = ['u', 'v', 'depth', 'rgb']
        scene_object_description = {}
        for car_name, description_dict in scene.original_cars_dict.items():
            filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors}
            scene_object_description[car_name] = filtered_description_dict
        deletion_car_names = deletion_agent.llm_finding_deletion(scene, task, scene_object_description)
        for car_name in deletion_car_names:
            scene.remove_car(car_name)

    def view_adjust_operation(self, scene, task, tech_agents):
        """ view adjust operation. 
        Participants: view_adjust_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        view_adjust_agent = tech_agents['view_adjust_agent']
        is_ego_motion = view_adjust_agent.llm_reasoning_ego_motion(scene, task)
        if is_ego_motion:
            start_frame_in_nerf, end_frame_in_nerf = view_adjust_agent.llm_view_motion_gen(scene, task)
            view_adjust_agent.func_generate_extrinsic(scene, start_frame_in_nerf, end_frame_in_nerf)
        else:
            delta_extrinsic = view_adjust_agent.llm_view_adjust(scene, task)
            view_adjust_agent.func_update_extrinsic(scene, delta_extrinsic)

    def put_back_deleted_operation(self, scene, task, tech_agents):
        """ put back deleted operation. 
        Participants: deletion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        deletion_agent = tech_agents['deletion_agent']
        valid_object_descriptors = ['u', 'v', 'depth', 'rgb']
        scene_object_description = {}
        for car_name, description_dict in scene.original_cars_dict.items():
            filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors}
            scene_object_description[car_name] = filtered_description_dict
        put_back_car_names = deletion_agent.llm_putting_back_deletion(scene, task, scene_object_description)
        for car_name in put_back_car_names:
            scene.removed_cars.remove(car_name)

    def revise_added_operation(self, scene, task, tech_agents):
        """ revised added vehicle 
        Participants: asset_select_agent, motion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        asset_select_agent = tech_agents['asset_select_agent']
        motion_agent = tech_agents['motion_agent']
        for added_car_name, added_car_info in scene.added_cars_dict.items():
            added_car_info['x'] = added_car_info['motion'][0][0]
            added_car_info['y'] = added_car_info['motion'][0][1]
        added_cars_short_dict = copy.deepcopy(scene.added_cars_dict)
        for added_car_name, added_car_info in added_cars_short_dict.items():
            added_car_info.pop('motion')
            if 'mode' in added_car_info:
                added_car_info.pop('mode')
                added_car_info.pop('distance_constraint')
                added_car_info.pop('distance_min_max')
                added_car_info.pop('need_placement_and_motion')
        modified_car_dict = asset_select_agent.llm_revise_added_cars(scene, task, added_cars_short_dict)
        for modified_car_name, modified_car_info in modified_car_dict.items():
            scene.added_cars_dict[modified_car_name]['color'] = modified_car_info['color']
            scene.added_cars_dict[modified_car_name]['type'] = modified_car_info['type']
            scene.added_cars_dict[modified_car_name]['need_placement_and_motion'] = False
            check_attributes = ['action', 'speed', 'direction', 'x', 'y']
            for attri in check_attributes:
                if scene.added_cars_dict[modified_car_name][attri] != modified_car_info[attri]:
                    scene.added_cars_dict[modified_car_name]['need_placement_and_motion'] = True
                    scene.added_cars_dict[modified_car_name][attri] = modified_car_info[attri]
            motion_agent.func_placement_and_motion_single_vehicle(scene, modified_car_name)

def visualize_poses(poses, camera_coord_axis_order='DRB', size=0.1):
    """
    Args:
        poses : numpy.ndarray
            shape [B, 3/4, 4]

        size : float
            size of axis

        camera_coord_axis_order : str
            https://zhuanlan.zhihu.com/p/593204605
            how camera coordinate's xyz related to the camera view
            For example, 'DRB' means x->down, y->right, z->back. 
            ======================
            OpenCV/Colmap: RDF
            LLFF: DRB
            OpenGL/NeRF: RUB
            Blender: RUB
            Mitsuba/Pytorch3D: LUF

    """
    try:
        camera_front = camera_view_dir(camera_coord_axis_order.index('F'), 1)
    except:
        camera_front = camera_view_dir(camera_coord_axis_order.index('B'), -1)
    try:
        camera_right = camera_view_dir(camera_coord_axis_order.index('R'), 1)
    except:
        camera_right = camera_view_dir(camera_coord_axis_order.index('L'), -1)
    try:
        camera_up = camera_view_dir(camera_coord_axis_order.index('U'), 1)
    except:
        camera_up = camera_view_dir(camera_coord_axis_order.index('D'), -1)
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]
    if poses.shape[1] == 3:
        pad_values = np.array([0, 0, 0, 1.0])
        poses = np.pad(poses, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
        poses[:, -1, :] = pad_values
    for pose in poses:
        axes = trimesh.creation.axis(transform=pose, axis_length=size)
        objects.append(axes)
        pos = pose[:3, 3]
        up_left = pos + camera_front.pn * size * pose[:3, camera_front.axis] + camera_up.pn * size * pose[:3, camera_up.axis] - camera_right.pn * size * pose[:3, camera_right.axis]
        up_right = pos + camera_front.pn * size * pose[:3, camera_front.axis] + camera_up.pn * size * pose[:3, camera_up.axis] + camera_right.pn * size * pose[:3, camera_right.axis]
        down_left = pos + camera_front.pn * size * pose[:3, camera_front.axis] - camera_up.pn * size * pose[:3, camera_up.axis] - camera_right.pn * size * pose[:3, camera_right.axis]
        down_right = pos + camera_front.pn * size * pose[:3, camera_front.axis] - camera_up.pn * size * pose[:3, camera_up.axis] + camera_right.pn * size * pose[:3, camera_right.axis]
        dir = (up_left + up_right + down_left + down_right) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-08)
        o = pos + dir * 2
        up_middle = (up_left + up_right) / 2
        segs = np.array([[pos, up_left], [pos, up_right], [pos, down_left], [pos, down_right], [up_left, up_right], [up_right, down_right], [down_right, down_left], [down_left, up_left], [pos, o], [pos, up_middle]])
        segs = trimesh.load_path(segs)
        objects.append(segs)
    trimesh.Scene(objects).show()

class Timer:

    def __init__(self):
        self.time = time.time()
        print('Start timing ... ')

    def print(self, message=''):
        print(f'\n--- {message} using time {time.time() - self.time:3f} ---\n')
        self.time = time.time()

def plot_feature(feature, channel, save_path, flag='', vmin=None, vmax=None, colorbar=True):
    """
    Args:
        feature : torch.tensor or np.ndarry
            suppose in shape [N, C, H, W]

        channel : int or list of int
            channel for ploting

        save_path : str
            save path for visualizing results.
    """
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()
    if isinstance(channel, int):
        channel = [channel]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    N, C, H, W = feature.shape
    for c in channel:
        for n in range(N):
            plt.imshow(feature[n, c], vmin=vmin, vmax=vmax)
            file_path = os.path.join(save_path, f'{flag}_agent_{n}_channel_{c}.png')
            if colorbar:
                plt.colorbar()
            plt.savefig(file_path, dpi=400)
            plt.close()
            print(f'Saving to {file_path}')

def evaluate_peak_intensity(visualization_path):
    """
    peak_intensity_error_percentage : float
       (abs(pred_int - gt_int) / gt_int) * 100%

    Args: 
    
        visualization_path : str
            visulization is result on testset. 
            e.g. mc_to_sky/logs/pred_hdr_pano_from_AvgMultiView_enhanced_elu_white_balance_adjust3/lightning_logs/version_0/visualization
            include 'xxxx_hdr_gt.exr', 'xxxx_hdr_pred.exr'
    
    Returns:
        max, min, mean, median of peak_intensity_error_percentage
    """
    hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, '*_hdr_pred.exr')))
    hdr_gt_files = sorted(glob.glob(os.path.join(visualization_path, '*_hdr_gt.exr')))
    pred_peak_illuminance_list = []
    gt_peak_illuminance_list = []
    peak_error_list = []
    assert len(hdr_pred_files) == len(hdr_gt_files)
    for pred_file, gt_file in zip(hdr_pred_files, hdr_gt_files):
        pred = imageio.imread(pred_file)
        gt = imageio.imread(gt_file)
        pred_illuminance = 0.2126 * pred[..., 0] + 0.7152 * pred[..., 1] + 0.0722 * pred[..., 2]
        gt_illuminance = 0.2126 * gt[..., 0] + 0.7152 * gt[..., 1] + 0.0722 * gt[..., 2]
        pred_peak_illuminance = np.max(pred_illuminance)
        gt_peak_illuminance = np.max(gt_illuminance)
        if np.isinf(gt_peak_illuminance):
            continue
        pred_peak_illuminance_log10 = np.log10(pred_peak_illuminance).clip(0, 100)
        gt_peak_illuminance_log10 = np.log10(gt_peak_illuminance).clip(0, 100)
        peak_error = np.abs(pred_peak_illuminance_log10 - gt_peak_illuminance_log10) / gt_peak_illuminance_log10
        peak_error_list.append(peak_error)
    print(visualization_path)
    print(f'{colored('min: ', 'green')} {np.min(peak_error_list)}')
    print(f'{colored('max: ', 'green')} {np.max(peak_error_list)}')
    print(f'{colored('mean: ', 'green')} {np.mean(peak_error_list)}')
    print(f'{colored('median: ', 'green')} {np.median(peak_error_list)}')
    return (np.min(peak_error_list), np.max(peak_error_list), np.mean(peak_error_list), np.median(peak_error_list))

def evaluate_peak_direction(visualization_path):
    """
    peak_direction_error_percentage : float
        angle of <pred_peak_dir, gt_peak_dir>

    Args: 
        visualization_path : str
            visulization is result on testset. 
            e.g. mc_to_sky/logs/Hold_Geoffroy_pred_hdr_pano_from_single/lightning_logs/version_0/visualization
            include 'xxxx_hdr_gt.exr', 'xxxx_hdr_pred.exr', ('xxxx_hdr_pred_rotated.exr')
    
    Returns:
        max, min, mean, median of peak_intensity_error_percentage
    """
    hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, '*_hdr_pred.exr')))
    hdr_gt_files = sorted(glob.glob(os.path.join(visualization_path, '*_ldr_input.png')))
    angular_error_list = []
    if len(glob.glob(os.path.join(visualization_path, '*_hdr_pred_rotated.exr'))) != 0:
        hdr_pred_files = sorted(glob.glob(os.path.join(visualization_path, '*_hdr_pred_rotated.exr')))
    assert len(hdr_pred_files) == len(hdr_gt_files)
    for pred_file, gt_file in zip(hdr_pred_files, hdr_gt_files):
        pred = imageio.imread(pred_file)
        gt = srgb_inv_gamma_correction(imageio.imread(gt_file) / 255)
        H, W, _ = pred.shape
        env_template = EnvironmentMap(H, 'skylatlong')
        pred_illuminance = 0.2126 * pred[..., 0] + 0.7152 * pred[..., 1] + 0.0722 * pred[..., 2]
        gt_illuminance = 0.2126 * gt[..., 0] + 0.7152 * gt[..., 1] + 0.0722 * gt[..., 2]
        max_index_pred = np.argmax(pred_illuminance, axis=None)
        max_index_pred_2d = np.unravel_index(max_index_pred, pred_illuminance.shape)
        peak_pred_v, peak_pred_u = max_index_pred_2d
        max_index_gt = np.argmax(gt_illuminance, axis=None)
        max_gt_illuminance = np.max(gt_illuminance)
        max_gt_illuminance_num = np.sum(gt_illuminance == max_gt_illuminance)
        if max_gt_illuminance_num > 15:
            continue
        max_index_gt_2d = np.unravel_index(max_index_gt, gt_illuminance.shape)
        peak_gt_v, peak_gt_u = max_index_gt_2d
        peak_pred_xyz = env_template.image2world(peak_pred_u / W, peak_pred_v / H)
        peak_gt_xyz = env_template.image2world(peak_gt_u / W, peak_gt_v / H)
        angular_error_cosine = np.dot(peak_pred_xyz / np.linalg.norm(peak_pred_xyz), peak_gt_xyz / np.linalg.norm(peak_gt_xyz))
        angular_error = np.degrees(np.arccos(angular_error_cosine))
        angular_error_list.append(angular_error)
    ic(len(angular_error_list))
    print(visualization_path)
    print(f'{colored('min: ', 'green')} {np.min(angular_error_list)}')
    print(f'{colored('max: ', 'green')} {np.max(angular_error_list)}')
    print(f'{colored('mean: ', 'green')} {np.mean(angular_error_list)}')
    print(f'{colored('median: ', 'green')} {np.median(angular_error_list)}')
    return (np.min(angular_error_list), np.max(angular_error_list), np.mean(angular_error_list), np.median(angular_error_list))

def srgb_inv_gamma_correction(gamma_corrected_image):
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)
    linear_image = np.where(gamma_corrected_image <= 0.04045, gamma_corrected_image / 12.92, ((gamma_corrected_image + 0.055) / 1.055) ** 2.4)
    return linear_image

def test_resolution():
    pano_image = imageio.imread('dataset/holicity_pano/2008-07/8heFyix0weuW7Kzd6A_BLg.jpg')
    H = pano_image.shape[0]
    for level in range(8):
        time_begin = time.time()
        pano_image_downsample = cv2.resize(pano_image, (H // 2 ** level * 2, H // 2 ** level))
        e = EnvironmentMap(pano_image_downsample, 'latlong')
        for j in range(3):
            outpath = f'crop_down{level}x_{j}.png'
            rotation_mat = rotation_matrix(azimuth=2 * np.pi / 10, elevation=0)
            e.rotate(rotation_mat)
            crop = e.project(vfov=60, ar=16 / 9, resolution=(640, 360), rotation_matrix=rotation_mat).astype(np.uint8)
            imageio.imsave(outpath, crop)
        print(f'Level {level} using time: {time.time() - time_begin}')

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

class SkyPred(pl.LightningModule):

    def __init__(self, hypes):
        super().__init__()
        self.hypes = hypes
        self.save_hyperparameters()
        self.latent_predictor = build_latent_predictor(hypes['model']['latent_predictor'])
        sky_model_core_method = hypes['model']['sky_model']['core_method']
        sky_model_core_method_ckpt_path = hypes['model']['sky_model']['ckpt_path']
        if sky_model_core_method == 'sky_model_enhanced':
            self.sky_model = SkyModelEnhanced.load_from_checkpoint(sky_model_core_method_ckpt_path)
        elif sky_model_core_method == 'sky_model':
            self.sky_model = SkyModel.load_from_checkpoint(sky_model_core_method_ckpt_path)
        self.ldr_recon_loss = build_loss(hypes['loss']['ldr_recon_loss'])
        self.hdr_recon_loss = build_loss(hypes['loss']['hdr_recon_loss'])
        self.peak_int_loss = build_loss(hypes['loss']['peak_int_loss'])
        self.peak_dir_loss = build_loss(hypes['loss']['peak_dir_loss'])
        self.latent_loss = build_loss(hypes['loss']['latent_loss'])
        self.fix_modules = hypes['model'].get('fix_modules', [])
        self.on_train_epoch_start()

    def decode_forward(self, latent_vector, peak_vector):
        return self.sky_model.decode_forward(latent_vector, peak_vector, peak_vector)

    def on_train_epoch_start(self):
        print(f'Module fixed in training: {self.fix_modules}.')
        for module in self.fix_modules:
            for p in eval(f'self.{module}').parameters():
                p.requires_grad_(False)
            eval(f'self.{module}').eval()

    def training_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_envmap_tensor, mask_envmap_tensor)
        ldr_recon_loss = self.ldr_recon_loss(srgb_gamma_correction_torch(hdr_skypano_pred), ldr_envmap_tensor, mask_envmap_tensor)
        latent_loss = self.latent_loss(latent_vector_pred, latent_vector_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[..., :3], peak_vector_gt[..., :3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[..., 3:], peak_vector_gt[..., 3:])
        loss = hdr_recon_loss + ldr_recon_loss + latent_loss + peak_dir_loss + peak_int_loss
        self.log('train_loss', loss)
        self.log('hdr_recon_loss', hdr_recon_loss)
        self.log('ldr_recon_loss', ldr_recon_loss)
        self.log('latent_loss', latent_loss)
        self.log('peak_dir_loss', peak_dir_loss)
        self.log('peak_int_loss', peak_int_loss)
        print(f'|| loss: {loss:.3f} || hdr_recon_loss: {hdr_recon_loss:.3f} || ldr_recon_loss: {ldr_recon_loss:.3f} ' + f'|| latent_loss: {latent_loss:.3f} || peak_dir_loss: {peak_dir_loss:.3f} || peak_int_loss: {peak_int_loss:.3f}')
        return loss

    def validation_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        mask_envmap_tensor = torch.gt(mask_envmap_tensor, 0.8)
        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)
        hdr_recon_loss = self.hdr_recon_loss(hdr_skypano_pred, hdr_envmap_tensor, mask_envmap_tensor)
        ldr_recon_loss = self.ldr_recon_loss(srgb_gamma_correction_torch(hdr_skypano_pred), ldr_envmap_tensor, mask_envmap_tensor)
        latent_loss = self.latent_loss(latent_vector_pred, latent_vector_gt)
        peak_dir_loss = self.peak_dir_loss(peak_vector_pred[..., :3], peak_vector_gt[..., :3])
        peak_int_loss = self.peak_int_loss(peak_vector_pred[..., 3:], peak_vector_gt[..., 3:])
        loss = hdr_recon_loss + ldr_recon_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        img_crops_tensor, peak_vector_gt, latent_vector_gt, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor = batch
        mask_envmap_tensor = torch.gt(mask_envmap_tensor, 0.8)
        peak_vector_pred, latent_vector_pred = self.latent_predictor(img_crops_tensor)
        hdr_skypano_pred, ldr_skypano_pred, _ = self.decode_forward(latent_vector_pred, peak_vector_pred)
        return_dict = {'hdr_skypano_pred': hdr_skypano_pred.permute(0, 2, 3, 1), 'ldr_skypano_pred': srgb_gamma_correction_torch(hdr_skypano_pred).permute(0, 2, 3, 1), 'hdr_skypano_gt': hdr_envmap_tensor.permute(0, 2, 3, 1), 'ldr_skypano_input': ldr_envmap_tensor.permute(0, 2, 3, 1), 'mask_env': mask_envmap_tensor.permute(0, 2, 3, 1), 'image_crops': img_crops_tensor.permute(0, 1, 3, 4, 2), 'batch_idx': batch_idx}
        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hypes['lr_schedule']['init_lr'])
        lr_scheduler = StepLR(optimizer=optimizer, step_size=self.hypes['lr_schedule']['decay_per_epoch'], gamma=self.hypes['lr_schedule']['decay_rate'])
        return ([optimizer], [lr_scheduler])

def set_hdri(hdri_path, rotation=None):
    """
    Args:
        hdri_path: str
            path to hdri
        rotation: list of float
            [rotate_x, rotate_y, rotate_z] rotate the HDRI. (rad)
            rotate_z (pos) will rotate the skydome clockwise

            By default, the HDRI is set to x-positive view.
    """
    C = bpy.context
    scn = C.scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()
    node_background = tree_nodes.new(type='ShaderNodeBackground')
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    node_environment.image = bpy.data.images.load(hdri_path)
    node_environment.location = (-300, 0)
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = (200, 0)
    links = node_tree.links
    link = links.new(node_environment.outputs['Color'], node_background.inputs['Color'])
    link = links.new(node_background.outputs['Background'], node_output.inputs['Surface'])
    if rotation is not None:
        node_map = tree_nodes.new('ShaderNodeMapping')
        node_map.location = (-500, 0)
        node_texcoor = tree_nodes.new('ShaderNodeTexCoord')
        node_texcoor.location = (-700, 0)
        link = links.new(node_texcoor.outputs['Generated'], node_map.inputs['Vector'])
        link = links.new(node_map.outputs['Vector'], node_environment.inputs['Vector'])
        if isinstance(rotation, list):
            node_map.inputs['Rotation'].default_value = rotation
        elif isinstance(rotation, str):
            if rotation == 'camera_view':
                camera_obj_name = 'Camera'
                camera = bpy.data.objects[camera_obj_name]
                camera.rotation_mode = 'XYZ'
                camera_rot_z = camera.rotation_euler.z
                print(camera.rotation_euler)
                node_map.inputs['Rotation'].default_value[2] = -camera_rot_z
                camera.rotation_mode = 'QUATERNION'
            else:
                raise 'This HDRI rotation is not implemented'
        else:
            raise 'This HDRI rotation is not implemented'

