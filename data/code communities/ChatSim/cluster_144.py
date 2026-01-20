# Cluster 144

def load_model_from_config(config, ckpt):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    global_step = pl_sd['global_step']
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return ({'model': model}, global_step)

def get_model(mode):
    path_conf, path_ckpt = download_models(mode)
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model

def download_models(mode):
    if mode == 'superresolution':
        url_conf = 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'
        url_ckpt = 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'
        path_conf = 'logs/diffusion/superresolution_bsr/configs/project.yaml'
        path_ckpt = 'logs/diffusion/superresolution_bsr/checkpoints/last.ckpt'
        download_url(url_conf, path_conf)
        download_url(url_ckpt, path_ckpt)
        path_conf = path_conf + '/?dl=1'
        path_ckpt = path_ckpt + '/?dl=1'
        return (path_conf, path_ckpt)
    else:
        raise NotImplementedError

def run(model, selected_path, task, custom_steps, resize_enabled=False, classifier_ckpt=None, global_step=None):
    example = get_cond(task, selected_path)
    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.0
    eta = 1.0
    make_progrow = True
    custom_shape = None
    height, width = example['image'].shape[1:3]
    split_input = height >= 128 and width >= 128
    if split_input:
        ks = 128
        stride = 64
        vqf = 4
        model.split_input_params = {'ks': (ks, ks), 'stride': (stride, stride), 'vqf': vqf, 'patch_distributed_vq': True, 'tie_braker': False, 'clip_max_weight': 0.5, 'clip_min_weight': 0.01, 'clip_max_tie_weight': 0.5, 'clip_min_tie_weight': 0.01}
    elif hasattr(model, 'split_input_params'):
        delattr(model, 'split_input_params')
    invert_mask = False
    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])
        logs = make_convolutional_sample(example, model, mode=mode, custom_steps=custom_steps, eta=eta, swap_mode=False, masked=masked, invert_mask=invert_mask, quantize_x0=False, custom_schedule=None, decode_interval=10, resize_enabled=resize_enabled, custom_shape=custom_shape, temperature=temperature, noise_dropout=0.0, corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid, make_progrow=make_progrow, ddim_use_x0_pred=ddim_use_x0_pred)
    return logs

def get_cond(mode, selected_path):
    example = dict()
    if mode == 'superresolution':
        up_f = 4
        visualize_cond_img(selected_path)
        c = Image.open(selected_path)
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
        c_up = rearrange(c_up, '1 c h w -> 1 h w c')
        c = rearrange(c, '1 c h w -> 1 h w c')
        c = 2.0 * c - 1.0
        c = c.to(torch.device('cuda'))
        example['LR_image'] = c
        example['image'] = c_up
    return example

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f'Loading model from {ckpt}')
        pl_sd = torch.load(ckpt, map_location='cpu')
        global_step = pl_sd['global_step']
    else:
        pl_sd = {'state_dict': None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd['state_dict'])
    return (model, global_step)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

