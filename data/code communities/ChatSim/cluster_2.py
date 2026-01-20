# Cluster 2

class ChatSim:

    def __init__(self, config):
        self.config = config
        self.scene = Scene(config['scene'])
        agents_config = config['agents']
        self.project_manager = ProjectManager(agents_config['project_manager'])
        self.asset_select_agent = AssetSelectAgent(agents_config['asset_select_agent'])
        self.deletion_agent = DeletionAgent(agents_config['deletion_agent'])
        self.foreground_rendering_agent = ForegroundRenderingAgent(agents_config['foreground_rendering_agent'])
        self.motion_agent = MotionAgent(agents_config['motion_agent'])
        self.view_adjust_agent = ViewAdjustAgent(agents_config['view_adjust_agent'])
        if agents_config['background_rendering_agent'].get('scene_representation', 'nerf') == 'nerf':
            self.background_rendering_agent = BackgroundRenderingAgent(agents_config['background_rendering_agent'])
        else:
            self.background_rendering_agent = BackgroundRendering3DGSAgent(agents_config['background_rendering_agent'])
        self.tech_agents = {'asset_select_agent': self.asset_select_agent, 'background_rendering_agent': self.background_rendering_agent, 'deletion_agent': self.deletion_agent, 'foreground_rendering_agent': self.foreground_rendering_agent, 'motion_agent': self.motion_agent, 'view_adjust_agent': self.view_adjust_agent}
        self.current_prompt = 'An empty prompt'

    def setup_init_frame(self):
        """Setup initial frame for ChatSim's reasoning and rendering.
        """
        if not os.path.exists(self.scene.init_img_path):
            print(f'{colored('[Note]', color='red', attrs=['bold'])} ', f'{colored('can not find init image, rendering it for the first time')}\n')
            self.background_rendering_agent.func_render_background(self.scene)
            imageio.imwrite(self.scene.init_img_path, self.scene.current_images[0])
        else:
            self.scene.current_images = [imageio.imread(self.scene.init_img_path)] * self.scene.frames

    def execute_llms(self, prompt):
        """Entry of ChatSim's reasoning.
        We perform multi-LLM reasoning for the user's prompt

        Input:
            prompt : str
                language prompt to ChatSim.
        """
        self.scene.setup_cars()
        self.current_prompt = prompt
        tasks = self.project_manager.decompose_prompt(self.scene, prompt)
        for task in tasks.values():
            print(f'{colored('[Performing Single Prompt]', on_color='on_blue', attrs=['bold'])} {colored(task, attrs=['bold'])}\n')
            self.project_manager.dispatch_task(self.scene, task, self.tech_agents)
        print(colored('scene.added_cars_dict', color='red', attrs=['bold']), end=' ')
        pprint.pprint(self.scene.added_cars_dict.keys())
        print(colored('scene.removed_cars', color='red', attrs=['bold']), end=' ')
        pprint.pprint(self.scene.removed_cars)

    def execute_funcs(self):
        """Entry of ChatSim's rendering functions
        We perform agent's functions following the self.scene's configuration.
        self.scene's configuration are updated in self.execute_llms()
        """
        self.background_rendering_agent.func_render_background(self.scene)
        self.deletion_agent.func_inpaint_scene(self.scene)
        self.asset_select_agent.func_retrieve_blender_file(self.scene)
        self.foreground_rendering_agent.func_blender_add_cars(self.scene)
        generate_video(self.scene, self.current_prompt)

def generate_video(scene, prompt, save_images=False):
    video_output_path = os.path.join(scene.output_dir, scene.logging_name)
    check_and_mkdirs(video_output_path)
    filename = prompt.replace(' ', '_')[:40]
    fps = scene.fps
    print(colored('[Compositing video]', 'blue', attrs=['bold']), 'start...')
    writer = imageio.get_writer(os.path.join(video_output_path, f'{filename}.mp4'), fps=fps)
    for frame in tqdm(scene.final_video_frames):
        writer.append_data(frame)
    writer.close()
    if save_images:
        check_and_mkdirs(os.path.join(video_output_path, f'{filename}'))
        for i, img in enumerate(scene.final_video_frames):
            imageio.imsave(os.path.join(video_output_path, f'{filename}/{i}.png'), img)
    if not scene.save_cache:
        scene.clean_cache()
    print(colored('[Compositing video]', 'blue', attrs=['bold']), 'done.')

def get_parser():
    parser = argparse.ArgumentParser(description='ChatSim argrument parser.')
    parser.add_argument('--config_yaml', '-y', type=str, default='config/waymo-1287.yaml', help='path to config file')
    parser.add_argument('--prompt', '-p', type=str, default='add a straight driving car in the scene', help='language prompt to ChatSim.')
    parser.add_argument('--simulation_name', '-s', type=str, default='demo', help='simulation experiment name.')
    args = parser.parse_args()
    return args

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def check_and_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def score_jnd_dataset(data_loader, func, name=''):
    """ Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    """
    ds = []
    gts = []
    for data in tqdm(data_loader.load_data(), desc=name):
        ds += func(data['p0'], data['p1']).data.cpu().numpy().tolist()
        gts += data['same'].cpu().numpy().flatten().tolist()
    sames = np.array(gts)
    ds = np.array(ds)
    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]
    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1 - sames_sorted)
    FNs = np.sum(sames_sorted) - TPs
    precs = TPs / (TPs + FPs)
    recs = TPs / (TPs + FNs)
    score = voc_ap(recs, precs)
    return (score, dict(ds=ds, sames=sames))

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def tqdm(x):
    return x

def benchmark():
    filename = sys.argv[1]
    img = Image.open(filename)
    data = np.array(img.getdata(), dtype=np.uint8)
    if len(data.shape) == 1:
        n_channels = 1
        reshape = (img.height, img.width)
    else:
        n_channels = min(data.shape[1], 3)
        data = data[:, :n_channels]
        reshape = (img.height, img.width, n_channels)
    data = data.reshape(reshape).astype(np.uint8)
    methods = [simplest_countless, quick_countless, quick_countless_xor, quickest_countless, stippled_countless, zero_corrected_countless, countless, downsample_with_averaging, downsample_with_max_pooling, ndzoom, striding]
    formats = {1: 'L', 3: 'RGB', 4: 'RGBA'}
    if not os.path.exists('./results'):
        os.mkdir('./results')
    N = 500
    img_size = float(img.width * img.height) / 1024.0 / 1024.0
    print('N = %d, %dx%d (%.2f MPx) %d chan, %s' % (N, img.width, img.height, img_size, n_channels, filename))
    print('Algorithm\tMPx/sec\tMB/sec\tSec')
    for fn in methods:
        print(fn.__name__, end='')
        sys.stdout.flush()
        start = time.time()
        for _ in tqdm(range(N), desc=fn.__name__, disable=True):
            result = fn(data)
        end = time.time()
        print('\r', end='')
        total_time = end - start
        mpx = N * img_size / total_time
        mbytes = N * img_size * n_channels / total_time
        print('%s\t%.3f\t%.3f\t%.2f' % (fn.__name__, mpx, mbytes, total_time))
        outimg = Image.fromarray(np.squeeze(result), formats[n_channels])
        outimg.save('./results/{}.png'.format(fn.__name__, 'PNG'))

def score_2afc_dataset(data_loader, func, name=''):
    """ Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    """
    d0s = []
    d1s = []
    gts = []
    for data in tqdm(data_loader.load_data(), desc=name):
        d0s += func(data['ref'], data['p0']).data.cpu().numpy().flatten().tolist()
        d1s += func(data['ref'], data['p1']).data.cpu().numpy().flatten().tolist()
        gts += data['judge'].cpu().numpy().flatten().tolist()
    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s < d1s) * (1.0 - gts) + (d1s < d0s) * gts + (d1s == d0s) * 0.5
    return (np.mean(scores), dict(d0s=d0s, d1s=d1s, gts=gts, scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    """ Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    """
    ds = []
    gts = []
    for data in tqdm(data_loader.load_data(), desc=name):
        ds += func(data['p0'], data['p1']).data.cpu().numpy().tolist()
        gts += data['same'].cpu().numpy().flatten().tolist()
    sames = np.array(gts)
    ds = np.array(ds)
    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]
    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1 - sames_sorted)
    FNs = np.sum(sames_sorted) - TPs
    precs = TPs / (TPs + FPs)
    recs = TPs / (TPs + FNs)
    score = voc_ap(recs, precs)
    return (score, dict(ds=ds, sames=sames))

def get_activations(files, model, batch_size=50, dims=2048, cuda=False, verbose=False, keep_size=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    if len(files) % batch_size != 0:
        print('Warning: number of images is not a multiple of the batch size. Some samples are going to be ignored.')
    if batch_size > len(files):
        print('Warning: batch size is bigger than the data size. Setting batch size to data size')
        batch_size = len(files)
    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, dims))
    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        t = transform if not keep_size else ToTensor()
        if isinstance(files[0], pathlib.PosixPath):
            images = [t(Image.open(str(f))) for f in files[start:end]]
        elif isinstance(files[0], Image.Image):
            images = [t(f) for f in files[start:end]]
        else:
            raise ValueError(f'Unknown data type for image: {type(files[0])}')
        batch = torch.stack(images)
        if cuda:
            batch = batch.cuda()
        pred = model(batch)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    if verbose:
        print(' done')
    return pred_arr

def benchmark():
    filename = sys.argv[1]
    img = Image.open(filename)
    data = np.array(img.getdata(), dtype=np.uint8)
    if len(data.shape) == 1:
        n_channels = 1
        reshape = (img.height, img.width)
    else:
        n_channels = min(data.shape[1], 3)
        data = data[:, :n_channels]
        reshape = (img.height, img.width, n_channels)
    data = data.reshape(reshape).astype(np.uint8)
    methods = [simplest_countless, quick_countless, quick_countless_xor, quickest_countless, stippled_countless, zero_corrected_countless, countless, downsample_with_averaging, downsample_with_max_pooling, ndzoom, striding]
    formats = {1: 'L', 3: 'RGB', 4: 'RGBA'}
    if not os.path.exists('./results'):
        os.mkdir('./results')
    N = 500
    img_size = float(img.width * img.height) / 1024.0 / 1024.0
    print('N = %d, %dx%d (%.2f MPx) %d chan, %s' % (N, img.width, img.height, img_size, n_channels, filename))
    print('Algorithm\tMPx/sec\tMB/sec\tSec')
    for fn in methods:
        print(fn.__name__, end='')
        sys.stdout.flush()
        start = time.time()
        for _ in tqdm(range(N), desc=fn.__name__, disable=True):
            result = fn(data)
        end = time.time()
        print('\r', end='')
        total_time = end - start
        mpx = N * img_size / total_time
        mbytes = N * img_size * n_channels / total_time
        print('%s\t%.3f\t%.3f\t%.2f' % (fn.__name__, mpx, mbytes, total_time))
        outimg = Image.fromarray(np.squeeze(result), formats[n_channels])
        outimg.save('./results/{}.png'.format(fn.__name__, 'PNG'))

def video2frames(video_path, frame_path):
    video = cv2.VideoCapture(video_path)
    os.makedirs(frame_path, exist_ok=True)
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    initial_img = None
    for idx in tqdm(range(frame_num), 'Extract frames'):
        success, image = video.read()
        if idx == 0:
            initial_img = image.copy()
        assert success, 'extract the {}th frame in video {} failed!'.format(idx, video_path)
        cv2.imwrite('{}/{:05d}.jpg'.format(frame_path, idx), image)
    return (fps, initial_img)

def frames2video(frames_list, video_path, fps=30, remove_tmp=False):
    if isinstance(frames_list, str):
        frames_list = glob(f'{frames_list}/*.jpg')
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in tqdm(frames_list, 'Export video'):
        if isinstance(frame, str):
            frame = imageio.imread(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imageio.core.util.Array(frame)
        writer.append_data(frame)
    writer.close()
    print(f'find video at {video_path}.')
    if remove_tmp and isinstance(frames_list, str):
        shutil.rmtree(frames_list)

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert('RGB'))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    mask = np.array(Image.open(mask).convert('L'))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = (1 - mask) * image
    batch = {'image': image, 'mask': mask, 'masked_image': masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch

def load_multi_files(data_archive):
    database = {key: [] for key in data_archive[0].files}
    for d in tqdm(data_archive, desc=f'Loading datapool from {len(data_archive)} individual files.'):
        for key in d.files:
            database[key].append(d[key])
    return database

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

class ForegroundRenderingAgent:

    def __init__(self, config):
        self.config = config
        self.blender_dir = config['blender_dir']
        self.blender_utils_dir = config['blender_utils_dir']
        self.skydome_hdri_dir = config['skydome_hdri_dir']
        self.skydome_hdri_idx = config['skydome_hdri_idx']
        self.use_surrounding_lighting = config['use_surrounding_lighting']
        self.is_wide_angle = config['nerf_config']['is_wide_angle']
        self.scene_name = config['nerf_config']['scene_name']
        self.f2nerf_dir = config['nerf_config']['f2nerf_dir']
        self.nerf_exp_name = config['nerf_config']['nerf_exp_name']
        self.f2nerf_config = config['nerf_config']['f2nerf_config']
        self.dataset_name = config['nerf_config']['dataset_name']
        self.nerf_exp_dir = os.path.join(self.f2nerf_dir, 'exp', self.scene_name, self.nerf_exp_name)
        nerf_output_foler_name = 'wide_angle_novel_images' if self.is_wide_angle else 'novel_images'
        self.nerf_novel_view_dir = os.path.join(self.nerf_exp_dir, nerf_output_foler_name)
        self.nerf_quiet_render = config['nerf_config']['nerf_quiet_render']
        self.estimate_depth = config['estimate_depth']
        if self.estimate_depth:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            self.depth_est_method = config['depth_est']['method']
            self.sam_checkpoint = config['depth_est']['SAM']['ckpt']
            self.sam_model_type = config['depth_est']['SAM']['model_type']
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint).cuda()
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def func_blender_add_cars(self, scene):
        """
        use blender to add cars for multiple frames. Static image is one frame.

        call self.blender_add_cars_single_frame in multi processing
        """
        check_and_mkdirs(os.path.join(scene.cache_dir, 'blender_npz'))
        check_and_mkdirs(os.path.join(scene.cache_dir, 'blender_output'))
        check_and_mkdirs(os.path.join(scene.cache_dir, 'blender_yaml'))
        check_and_mkdirs(os.path.join(scene.cache_dir, 'spatial_varying_hdri'))
        output_path = os.path.join(scene.cache_dir, 'blender_output')
        if len(scene.added_cars_dict) > 0:
            scene.check_added_car_static()
            real_render_frames = 1 if scene.add_car_all_static else scene.frames
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Start rendering {real_render_frames} images.')
            print(f'see the log in {os.path.join(scene.cache_dir, 'rendering_log')} if save_cache is enabled')
            background_depth_list = []
            if self.estimate_depth:
                real_update_frames = scene.frames if scene.is_ego_motion else 1
                if self.depth_est_method == 'SAM':
                    background_depth_list = self.update_depth_batch_SAM(scene, scene.current_images[:real_update_frames])
                else:
                    raise NotImplementedError
                print(f'{colored('[Depth Estimation]', 'cyan', attrs=['bold'])} Finish depth estimation {real_update_frames} images.')
            print('preparing input files for blender rendering')
            for frame_id in tqdm(range(real_render_frames)):
                self.func_blender_add_cars_prepare_files_single_frame(scene, frame_id, background_depth_list)
            print(f'start rendering in parallel, process number is {scene.multi_process_num}.')
            print('This may take a few minutes. To speed up the foreground rendering, you can lower the `frames` number or render not-wide images.')
            print('If you find the results are incomplete or missing, that may due to OOM. You can reduce the multi_process_num in config yaml.')
            print('You can also check the log file for debugging with `save_cache` enabled in the yaml.')
            self.func_parallel_blender_rendering(scene)
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Finish rendering {real_render_frames} images.')
            for frame_id in range(real_render_frames, scene.frames):
                assert real_render_frames == 1
                source_blender_output_folder = f'{output_path}/0'
                target_blender_output_folder = f'{output_path}/{frame_id}'
                shutil.copytree(source_blender_output_folder, target_blender_output_folder, dirs_exist_ok=True)
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Copying Remaining {scene.frames - real_render_frames} images.')
            video_frames = []
            for frame_id in range(scene.frames):
                video_frame_file = os.path.join(scene.cache_dir, 'blender_output', str(frame_id), 'RGB_composite.png')
                img = imageio.imread(video_frame_file)
                video_frames.append(img)
        else:
            video_frames = scene.current_inpainted_images
        scene.final_video_frames = video_frames

    def func_blender_add_cars_prepare_files_single_frame(self, scene, frame_id, background_depth_list):
        np.savez(os.path.join(scene.cache_dir, 'blender_npz', f'{frame_id}.npz'), H=scene.height, W=scene.width, focal=scene.focal, rgb=scene.current_inpainted_images[frame_id], depth=background_depth_list[frame_id] if len(background_depth_list) > 0 else 1000, extrinsic=transform_nerf2opencv_convention(scene.current_extrinsics[frame_id]))
        car_list_for_blender = []
        for car_name, car_info in scene.added_cars_dict.items():
            car_blender_file = car_info['blender_file']
            car_list_for_blender.append({'new_obj_name': car_name, 'blender_file': car_blender_file, 'insert_pos': [car_info['motion'][frame_id, 0].tolist(), car_info['motion'][frame_id, 1].tolist(), 0], 'insert_rot': [0, 0, car_info['motion'][frame_id, 2].tolist()], 'model_obj_name': 'Car', **({'target_color': {'material_key': 'car_paint', 'color': [i / 255 for i in car_info['color']] + [1]}} if car_info['color'] != 'default' else {})})
        yaml_path = os.path.join(scene.cache_dir, 'blender_yaml', f'{frame_id}.yaml')
        output_path = os.path.join(scene.cache_dir, 'blender_output')
        skydome_hdri_path = os.path.join(self.skydome_hdri_dir, self.scene_name, f'{self.skydome_hdri_idx}.exr')
        final_hdri_path = skydome_hdri_path
        if self.use_surrounding_lighting:
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Generating Spatial Varying HDRI.')
            assert len(scene.added_cars_dict) == 1
            car_info = list(scene.added_cars_dict.values())[0]
            insert_x = car_info['motion'][frame_id, 0].tolist()
            insert_y = car_info['motion'][frame_id, 1].tolist()
            generate_rays(insert_x, insert_y, scene.ext_int_path, self.nerf_exp_dir)
            current_dir = os.getcwd()
            os.chdir(self.f2nerf_dir)
            print(f'{colored('[Mc-NeRF]', 'red', attrs=['bold'])} Generating Panorama.')
            render_command = f'python scripts/run.py                                     --config-name={self.f2nerf_config} dataset_name={self.dataset_name}                                     case_name={self.scene_name}                                     exp_name={self.nerf_exp_name}                                     mode=render_panorama_shutter                                     is_continue=true                                     +work_dir={os.getcwd()}'
            if self.nerf_quiet_render:
                render_command += ' > /dev/null 2>&1'
            os.system(render_command)
            os.chdir(current_dir)
            nerf_last_trans_file = os.path.join(self.nerf_exp_dir, 'last_trans.pt')
            nerf_panorama_dir = os.path.join(self.nerf_exp_dir, 'panorama')
            nerf_panorama_pngs = os.listdir(nerf_panorama_dir)
            assert len(nerf_panorama_pngs) == 1
            nerf_panorama_pt_file = os.path.join(self.nerf_exp_dir, 'nerf_panorama.pt')
            arbitray_H = 128
            sky_mask = np.zeros((arbitray_H, arbitray_H * 4, 3))
            nerf_env_panorama = torch.jit.load(nerf_panorama_pt_file).state_dict()['0'].cpu().numpy()
            nerf_last_trans = torch.jit.load(nerf_last_trans_file).state_dict()['0'].cpu().numpy()
            pure_sky_hdri_path = skydome_hdri_path.replace('.exr', '_sky.exr')
            sky_dome_panorama = imageio.imread(pure_sky_hdri_path)
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Merging HDRI')
            blending_panorama = blending_hdr_sky(nerf_env_panorama, sky_dome_panorama, nerf_last_trans, sky_mask)
            nerf_env_panorama_gamma_corrected = (srgb_gamma_correction(nerf_env_panorama) * 255).astype(np.uint8)
            sky_dome_panorama_gamma_corrected = (srgb_gamma_correction(sky_dome_panorama) * 255).astype(np.uint8)
            blending_hdr_sky_gamma_corrected = (srgb_gamma_correction(blending_panorama) * 255).astype(np.uint8)
            final_hdri_path = os.path.join(scene.cache_dir, 'spatial_varying_hdri', f'{frame_id}.exr')
            imageio.imwrite(final_hdri_path.replace('.exr', '_env.png'), nerf_env_panorama_gamma_corrected)
            imageio.imwrite(final_hdri_path.replace('.exr', '_sky.png'), sky_dome_panorama_gamma_corrected)
            imageio.imwrite(final_hdri_path.replace('.exr', '_blending.png'), blending_hdr_sky_gamma_corrected)
            sky_H, sky_W, _ = blending_panorama.shape
            blending_panorama_full = np.zeros((sky_H * 2, sky_W, 3))
            blending_panorama_full[:sky_H] = blending_panorama
            imageio.imwrite(final_hdri_path, blending_panorama_full.astype(np.float32))
            print(f'{colored('[Blender]', 'magenta', attrs=['bold'])} Finish Merging HDRI')
        blender_dict = {'render_name': str(frame_id), 'output_dir': output_path, 'scene_file': os.path.join(scene.cache_dir, 'blender_npz', f'{frame_id}.npz'), 'hdri_file': final_hdri_path, 'render_downsample': 2, 'cars': car_list_for_blender, 'depth_and_occlusion': scene.depth_and_occlusion, 'backup_hdri': scene.backup_hdri}
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=blender_dict, stream=f, allow_unicode=True)

    def func_compose_with_new_depth_single_frame(self, scene, frame_id):
        output_path = os.path.join(scene.cache_dir, 'blender_output')
        background_image = imageio.imread(os.path.join(output_path, str(frame_id), 'backup', 'RGB.png'))
        depth_map = np.load(f'{output_path}/{frame_id}/depth/background_depth.npy')
        sys.path.append(os.path.join(self.blender_utils_dir, 'postprocess'))
        import compose
        compose.compose(os.path.join(output_path, str(frame_id)), background_image, depth_map, 2)

    def func_parallel_blender_rendering(self, scene):
        multi_process_num = scene.multi_process_num
        log_dir = os.path.join(scene.cache_dir, 'rendering_log')
        check_and_mkdirs(os.path.join(scene.cache_dir, 'rendering_log'))
        frames = scene.frames
        segment_length = frames // multi_process_num
        processes = []
        for i in range(multi_process_num):
            start_frame = i * segment_length
            end_frame = (i + 1) * segment_length if i < multi_process_num - 1 else frames
            log_file = os.path.join(log_dir, f'{i}.txt')
            command = f'{self.blender_dir} -b --python {self.blender_utils_dir}/main_multicar.py -- {os.path.join(scene.cache_dir, 'blender_yaml')} -- {start_frame} -- {end_frame} > {log_file}'
            with open(log_file, 'w') as f:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append(process)
        for process in processes:
            stdout, stderr = process.communicate()

    def get_sparse_depth_from_LiDAR(self, scene, frame_id):
        extrinsic_opencv = transform_nerf2opencv_convention(scene.current_extrinsics[frame_id])
        pointcloud_world = np.concatenate((scene.pcd, np.ones((scene.pcd.shape[0], 1))), axis=1).T
        pointcloud_camera = (np.linalg.inv(extrinsic_opencv) @ pointcloud_world)[:3]
        pointcloud_image = (scene.intrinsics @ pointcloud_camera)[:2] / pointcloud_camera[2:3]
        z_positive = pointcloud_camera[2] > 0
        valid_points = (pointcloud_image[0] > 0) & (pointcloud_image[0] < scene.width) & (pointcloud_image[1] > 0) & (pointcloud_image[1] < scene.height) & z_positive
        pointcloud_image_valid = pointcloud_image[:, valid_points]
        valid_u_coord = pointcloud_image_valid[0].astype(np.int32)
        valid_v_coord = pointcloud_image_valid[1].astype(np.int32)
        sparse_depth_map = np.zeros((scene.height, scene.width))
        sparse_depth_map[valid_v_coord, valid_u_coord] = pointcloud_camera[2, valid_points]
        return sparse_depth_map

    def update_depth_batch_SAM(self, scene, image_list):
        """
        update depth batch use [SAM] + [LiDAR projection correction] to get instance-level depth

        Args:
            image_list : list of np.ndarray, len = 1 or scene.frames
                image is [H, W, 3] shape

        Returns:
            overlap_depth_list : list of np.array, len = 1 or scene.frames
                depth is [H, W] shape
        """
        real_update_frames = len(image_list)
        overlap_depth_list = []
        for frame_id in range(real_update_frames):
            output_path = os.path.join(scene.cache_dir, 'blender_output')
            rendered_car_mask = imageio.imread(f'{output_path}/{frame_id}/mask/vehicle_and_shadow0001.exr')
            rendered_car_mask = cv2.resize(rendered_car_mask, (scene.current_inpainted_images[frame_id].shape[1], scene.current_inpainted_images[frame_id].shape[0]))
            rendered_car_mask = rendered_car_mask[..., 0] > 20 / 255
            masks = self.mask_generator.generate(scene.current_inpainted_images[frame_id])
            num_masks = len(masks)
            import itertools
            mask_pairs = list(itertools.permutations(range(num_masks), 2))
            valid_mask_idx = np.ones(num_masks, dtype=bool)
            for pair in mask_pairs:
                mask_1 = masks[pair[0]]
                mask_2 = masks[pair[1]]
                if (mask_1['segmentation'] & mask_2['segmentation']).sum() > 0:
                    if mask_1['area'] < mask_2['area']:
                        valid_mask_idx[pair[0]] = False
                    else:
                        valid_mask_idx[pair[1]] = False
            idx = np.where(valid_mask_idx == True)[0]
            masks = [masks[i] for i in idx]
            sparse_depth_map = self.get_sparse_depth_from_LiDAR(scene, frame_id)
            sparse_depth_mask = sparse_depth_map != 0
            overlap_depth = np.ones((scene.height, scene.width)) * 500
            for i in range(len(masks)):
                intersection_area = masks[i]['segmentation'] & rendered_car_mask
                if intersection_area.sum() > 0:
                    intersection_area_with_depth = intersection_area & sparse_depth_mask
                    if intersection_area_with_depth.sum() > 0 and intersection_area_with_depth.sum() > 10:
                        avg_depth = sparse_depth_map[intersection_area_with_depth].mean()
                        min_depth = sparse_depth_map[intersection_area_with_depth].min()
                        max_depth = sparse_depth_map[intersection_area_with_depth].max()
                        median_depth = np.median(sparse_depth_map[intersection_area_with_depth])
                        overlap_depth[intersection_area] = avg_depth
            overlap_depth_list.append(overlap_depth.astype(np.float32))
        return overlap_depth_list

def main():
    args = get_parser()
    hypes = read_yaml(args.config)
    train_conf = hypes['train_conf']
    train_set = build_dataset(hypes, split='train')
    valid_set = build_dataset(hypes, split='val')
    train_loader = DataLoader(train_set, batch_size=train_conf['batch_size'], shuffle=True, num_workers=24, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=train_conf['batch_size'], shuffle=False, num_workers=24, pin_memory=True)
    if args.ckpt_path and (not args.load_weight_only):
        exp_dir = args.ckpt_path.split('lightning_logs')[0]
    else:
        exp_dir = get_exp_dir(hypes['exp_name'])
        dump_yaml(hypes, exp_dir)
    backup_script(exp_dir)
    model = build_model(hypes)
    if args.load_weight_only:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
        args.ckpt_path = None
    checkpoint_callback = get_callback()
    trainer = pl.Trainer(default_root_dir=exp_dir, accelerator=train_conf['accelerator'], devices=train_conf['device_num'], max_epochs=train_conf['epoch'], check_val_every_n_epoch=train_conf['check_val_every_n_epoch'], log_every_n_steps=train_conf['log_every_n_steps'], callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.ckpt_path)

def build_dataset(hypes, split):
    dataset_args = hypes['dataset']
    dataset_cls = eval(dataset_args['name'])
    return dataset_cls(dataset_args, split)

def get_exp_dir(expname):
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime(f'{expname}_%m%d_%H%M%S')
    root_dir = os.getcwd()
    root_dir_log = os.path.join(root_dir, 'mc_to_sky/logs')
    if not os.path.exists(root_dir_log):
        os.mkdir(root_dir_log)
    exp_dir = os.path.join(root_dir_log, current_time_str)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    return exp_dir

def dump_yaml(data, savepath):
    with open(os.path.join(savepath, 'config.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def backup_script(full_path, folders_to_save=['model', 'data_utils', 'utils', 'loss', 'tools']):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    current_path = os.path.dirname(__file__)
    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder, dirs_exist_ok=True)

def build_model(hypes, return_cls=False):
    model_args = hypes['model']
    model_name = model_args['name']
    model_filename = 'mc_to_sky.model.' + model_name
    model_lib = importlib.import_module(model_filename)
    model_cls = None
    target_model_name = model_name.replace('_', '')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model_cls = cls
    if return_cls:
        return model_cls
    model = model_cls(hypes)
    return model

def get_callback():
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.2f}', save_top_k=1, save_last=True, mode='min')
    return checkpoint_callback

def main():
    args = get_parser()
    hypes = read_yaml(args.config)
    model = build_model(hypes, return_cls=True).load_from_checkpoint(args.ckpt_path, hypes=hypes).to('cuda')
    model.eval()
    model_name = hypes['model']['name']
    skip = hypes['dataset']['view_setting']['view_num']
    skip = 3
    all_waymo = args.waymo_scenes_dir
    scenes = os.listdir(all_waymo)
    for scene in tqdm(scenes):
        scene_image_dir = os.path.join(all_waymo, scene, 'images')
        scene_output_dir = os.path.join(args.output_dir, scene)
        check_and_mkdirs(scene_output_dir)
        filename_list = sorted(os.listdir(scene_image_dir))
        for idx in range(0, len(filename_list), skip):
            input_W = hypes['dataset']['view_setting']['camera_W'] // hypes['dataset']['view_setting']['downsample_for_crop']
            input_H = hypes['dataset']['view_setting']['camera_H'] // hypes['dataset']['view_setting']['downsample_for_crop']
            image_paths = [os.path.join(scene_image_dir, filename_list[idx + ii]) for ii in range(skip)]
            images = [imageio.imread(image_path).astype(np.float32) / 255.0 for image_path in image_paths]
            images_crop = [cv2.resize(image, (input_W, input_H)) for image in images]
            images_input = [torch.from_numpy(image_crop).permute(2, 0, 1).unsqueeze(0).to('cuda') for image_crop in images_crop]
            inputs = torch.stack(images_input, dim=1)
            hdr_skypano = infer_sky(model, inputs)
            imageio.imwrite(os.path.join(scene_output_dir, image_paths[0].split('/')[-1].replace('.png', '_sky.exr')), hdr_skypano)
            hdr_fullpano = np.zeros((hdr_skypano.shape[0] * 2, hdr_skypano.shape[1], 3), dtype=np.float32)
            hdr_fullpano[:hdr_skypano.shape[0]] = hdr_skypano
            imageio.imwrite(os.path.join(scene_output_dir, image_paths[0].split('/')[-1].replace('.png', '.exr')), hdr_fullpano)
            SAVE_FULL_NPZ = False
            if SAVE_FULL_NPZ:
                poses_bounds = image_paths[0].split('images')[0] + 'poses_bounds.npy'
                waymo_ext_int = np.load(poses_bounds)[:, :15].reshape(-1, 3, 5)
                waymo_ext = waymo_ext_int[idx, :3, :4]
                waymo_ext_opencv = np.stack([waymo_ext[:, 1], waymo_ext[:, 0], -waymo_ext[:, 2], waymo_ext[:, 3]], axis=-1)
                waymo_ext_pad = np.identity(4)
                waymo_ext_pad[:3, :4] = waymo_ext_opencv
                waymo_int = waymo_ext_int[idx, :3, 4]
                print(os.path.join(scene_output_dir, image_paths[0].split('/')[-1].replace('png', 'npz')))
                np.savez(os.path.join(scene_output_dir, image_paths[0].split('/')[-1].replace('png', 'npz')), H=int(waymo_int[0]), W=int(waymo_int[1]), focal=waymo_int[2], rgb=imageio.imread(image_paths[0]), depth=np.full((int(waymo_int[0]), int(waymo_int[1])), 10000.0), extrinsic=waymo_ext_pad)

def infer_sky(sky_pred, img_crop):
    """
    Args:
        img_crop: torch.tensor
            [1, 3, H, W], range 0-1
    """
    with torch.no_grad():
        peak_vector, latent_vector = sky_pred.latent_predictor(img_crop)
        hdr_skypano, ldr_skypano_pred, _ = sky_pred.decode_forward(latent_vector, peak_vector)
        hdr_skypano = hdr_skypano.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.float32)
        return hdr_skypano

def adjust_rotation(image, azimuth=None):
    if azimuth is None:
        azimuth = np.random.rand() * 2 * np.pi
    envmap = EnvironmentMap(image, 'skylatlong')
    envmap.rotate(dcm=rotation_matrix(azimuth=azimuth, elevation=0))
    return envmap.data

def infer_and_save(sky_model, ldr_skypano, save_dir, filename):
    """
    Args:
        ldr_skypano: torch.tensor
            [1, 3, H, W]
    """
    with torch.no_grad():
        peak_vector, latent_vector = sky_model.encode_forward(ldr_skypano)
        hdr_skypano, ldr_skypano, _ = sky_model.decode_forward(latent_vector, peak_vector, peak_vector)
        sample_pseudo_gt = {'peak_vector': peak_vector.squeeze().cpu().numpy(), 'latent_vector': latent_vector.squeeze().cpu().numpy(), 'hdr_skypano': hdr_skypano.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.float32)}
        np.savez(os.path.join(save_dir, filename.replace('.jpg', '.npz')), **sample_pseudo_gt)
        hdr_skypano = hdr_skypano.permute(0, 2, 3, 1).squeeze().cpu().numpy().astype(np.float32)
        return hdr_skypano

def resize_all(source_dir='dataset/holicity_pano', target_dir='dataset/holicity_pano_resized_800'):
    record_dates = os.listdir(source_dir)
    for record_date in tqdm(record_dates):
        source_date_dir = os.path.join(source_dir, record_date)
        target_date_dir = os.path.join(target_dir, record_date)
        if not os.path.exists(target_date_dir):
            os.mkdir(target_date_dir)
        pano_filenames = os.listdir(source_date_dir)
        for pano_filename in pano_filenames:
            image = imageio.imread(os.path.join(source_date_dir, pano_filename))
            image_resize = cv2.resize(image, (1600, 800))
            imageio.imsave(os.path.join(target_date_dir, pano_filename), image_resize)

def resize_sky(source_dir='dataset/holicity_pano_resized_800', target_dir='dataset/holicity_pano_sky_resized_64'):
    record_dates = os.listdir(source_dir)
    for record_date in tqdm(record_dates):
        source_date_dir = os.path.join(source_dir, record_date)
        target_date_dir = os.path.join(target_dir, record_date)
        if not os.path.exists(target_date_dir):
            os.mkdir(target_date_dir)
        pano_filenames = os.listdir(source_date_dir)
        for pano_filename in pano_filenames:
            image = imageio.imread(os.path.join(source_date_dir, pano_filename))
            image_resize = cv2.resize(image[:image.shape[0] // 2, :, :], (256, 64))
            imageio.imsave(os.path.join(target_date_dir, pano_filename), image_resize)

def crop_pano_single(ns, record_date):
    print(f'In this process, we handle {record_date}')
    source_date_dir = os.path.join(ns['source_dir'], record_date)
    pano_filenames = os.listdir(source_date_dir)
    for azimuth_deg in range(0, 360, ns['degree_interval']):
        check_and_mkdirs(os.path.join(ns['target_dir'], str(azimuth_deg), record_date))
    for pano_filename in ns['selected_sample_dict'][record_date]:
        azimuth_deg = range(0, 360, ns['degree_interval'])[-1]
        pass_flag = os.path.exists(os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename))
        if pass_flag:
            continue
        image = cv2.imread(os.path.join(source_date_dir, pano_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
        pano_envmap = EnvironmentMap(image, 'latlong')
        for azimuth_deg in range(0, 360, ns['degree_interval']):
            if os.path.exists(os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename)):
                continue
            azimuth_rad = np.radians(azimuth_deg)
            rotation_mat_i = rotation_matrix(azimuth=azimuth_rad, elevation=0)
            img_crop = pano_envmap.project(vfov=ns['camera_vfov'], ar=ns['aspect_ratio'], resolution=(ns['crop_W'], ns['crop_H']), rotation_matrix=rotation_mat_i)
            img_crop = (img_crop * 255).astype(np.uint8)
            imageio.imsave(os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename), img_crop)
            print(f'save to {os.path.join(ns['target_dir'], str(azimuth_deg), record_date, pano_filename)}')

def crop_pano(source_dir='dataset/holicity_pano', target_dir='dataset/holicity_crop_multiview', selected_sample_json='dataset/holicity_meta_info/selected_sample.json', camera_H=1280, camera_W=1920, focal=2088.465, downsample_for_crop=4, degree_interval=45, multiprocess=-1):
    crop_H = camera_H // downsample_for_crop
    crop_W = camera_W // downsample_for_crop
    camera_vfov = np.degrees(np.arctan2(camera_H, camera_W)) * 2
    aspect_ratio = camera_W / camera_H
    with open(selected_sample_json) as f:
        selected_sample = json.load(f)
    sample_dict = {}
    for selected_sample_name in selected_sample:
        date, filename = selected_sample_name.split('/')
        if date in sample_dict:
            sample_dict[date].append(filename)
        else:
            sample_dict[date] = [filename]
    info_dict = {}
    info_dict['crop_H'] = crop_H
    info_dict['crop_W'] = crop_W
    info_dict['camera_vfov'] = camera_vfov
    info_dict['aspect_ratio'] = aspect_ratio
    info_dict['degree_interval'] = degree_interval
    info_dict['source_dir'] = source_dir
    info_dict['target_dir'] = target_dir
    info_dict['selected_sample_dict'] = sample_dict
    record_dates = sorted(sample_dict.keys())
    if multiprocess <= 0:
        for record_date in record_dates:
            crop_pano_single(info_dict, record_date)
    else:
        pool = Pool(multiprocess)
        for record_date in record_dates:
            pool.apply_async(func=crop_pano_single, args=(info_dict, record_date))
        pool.close()
        pool.join()

