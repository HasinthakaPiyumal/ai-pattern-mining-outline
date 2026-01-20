# Cluster 4

def calc_fid(image_path, ref_path, num_expected=50000, seed=0, batch=64):
    """Calculate FID for a given set of images."""
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))
    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0('Calculating FID...')
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
        print(f'{fid:g}')
    torch.distributed.barrier()
    boxx.mg()
    if dist.get_rank() == 0:
        return dict(fid=fid, mu=mu, sigma=sigma)

def calculate_inception_stats(image_path, num_expected=None, seed=0, max_batch_size=64, num_workers=3, prefetch_factor=2, device=torch.device('cuda')):
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=dist.get_rank() == 0) as f:
        detector_net = pickle.load(f).to(device)
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=dist.get_rank() != 0):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    boxx.mg()
    return (mu.cpu().numpy(), sigma.cpu().numpy())

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path', help='Destination .npz file', metavar='NPZ', type=str, required=True)
@click.option('--batch', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    print('Warning!!! Creating fid ref often fails mysteriously. Recommendations:\n1. Set `--batch` to a small value that is a divisor of `len(dataset)`\n2. Try multiple times')
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()
    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)
    torch.distributed.barrier()
    dist.print0('Done.')

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--outdir', help='Where to save the output images', metavar='DIR', type=str)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-99', show_default=True)
@click.option('--subdirs', help='Create subdirectory for every 1000 seeds', is_flag=True)
@click.option('--class', 'class_idx', help='Class label  [default: random]', metavar='INT', type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min', help='Lowest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', help='Highest noise level  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min', help='Stoch. min noise level', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max', help='Stoch. max noise level', metavar='FLOAT', type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise', help='Stoch. noise inflation', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--solver', help='Ablate ODE solver', metavar='euler|heun', type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization', help='Ablate time step discretization {t_i}', metavar='ddn|vp|ve|iddpm|edm', type=click.Choice(['ddn', 'vp', 've', 'iddpm', 'edm']))
@click.option('--schedule', help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear', type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling', help='Ablate signal scaling s(t)', metavar='vp|none', type=click.Choice(['vp', 'none']))
@click.option('--learn-res', help='learn_residual in SDDNOutput', metavar='BOOL', type=bool, default=None, show_default=True)
@click.option('--skip-exist', help='skip-exist', metavar='BOOL', type=bool, default=None, show_default=True)
@click.option('--sampler', help='Guided sampler', default=None, type=click.Choice(['none', 'train', 'test', 'class', 'xflip', 'entropy']))
@click.option('--markov', help='Markov Sampling [pt_path, 1,0]', metavar='PATH|INT', type=str, default=None)
@click.option('--debug', help='debug mode', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--fid_ref', help='fid ref path e.g. fid-refs/cifar10-32x32.npz', metavar='PATH', type=str, default=None, show_default=True)
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), learn_res=None, skip_exist=True, sampler=None, markov=None, **sampler_kwargs):
    'Generate random images using the techniques described in the paper\n    "Elucidating the Design Space of Diffusion-Based Generative Models".\n\n    Examples:\n\n    \x08\n    # Generate 64 images and save them as out/*.png\n    python generate.py --outdir=out --seeds=0-63 --batch=64 \\\n        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl\n\n    \x08\n    # Generate 1024 images using 2 GPUs\n    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\\n        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl\n    ' + '\n    # Doc string for DDN:\n    If len(seeds) > 50000, will eval FID.\n    '
    is_s3 = network_pkl.startswith('s3://')
    if outdir is None:
        if is_s3:
            outdir = '/run/generate'
        else:
            outdir = os.path.abspath(os.path.join(network_pkl, '..', 'generate'))
        os.makedirs(outdir, exist_ok=True)
    visp = (network_pkl + '$').replace('.pkl$', '-vis.png').replace('.pt$', '-vis.png') if outdir.endswith('/generate') else os.path.abspath(outdir) + '-vis.png'
    eval_dir = visp.replace('-vis.png', '')
    sampler_cmd = '' if sampler is None or sampler == 'none' else sampler
    sampler_prefix = sampler_cmd and f'sampler.{sampler_cmd}-'
    fid_path = os.path.join(eval_dir, sampler_prefix + 'fid.json')
    if skip_exist is None:
        skip_exist = len(seeds) in [100, 50000]
    if markov:
        markov = int(markov) if len(markov) == 1 else markov
        if markov:
            assert markov != 1, 'NotImplement!'
            from zero_condition.markov_sampler import MarkovSampler
            markov_sampler = MarkovSampler(markov)
    dist.init()
    if skip_exist and (os.path.exists(visp) and len(seeds) == 100) or (os.path.exists(fid_path) and len(seeds) == 50000):
        if torch.distributed.get_rank() == 0:
            print('Vis exists:', visp)
        return
    dirr = os.path.dirname(network_pkl)
    training_options_json = os.path.join(dirr, 'training_options.json')
    if os.path.exists(training_options_json):
        train_kwargs = boxx.loadjson(training_options_json)
        boxx.cf.kwargs = train_kwargs.get('kwargs', {})
        if learn_res is None:
            learn_res = train_kwargs.get('kwargs', {}).get('learn_res', 'learn.res' in network_pkl)
        sddn.DiscreteDistributionOutput.learn_residual = learn_res
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=dist.get_rank() == 0) as f:
        if network_pkl.endswith('.pkl'):
            net = pickle.load(f)['ema'].to(device)
        elif network_pkl.endswith('.pt'):
            net = torch.load(f)['net'].to(device)
            net = net.eval()
    boxx.mg()
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    idx_ks_list = []
    if sampler_cmd:
        from zero_condition.main import ReconstructionDatasetSampler, CifarSampler, BatchedGuidedSampler
        if sampler_cmd == 'class':
            sampler = CifarSampler(None)
            print('CifarSampler!!!')
        if sampler_cmd == 'entropy':
            sampler = CifarSampler(entropy=True)
            print('CifarSampler(entropy=True)!!!')
        if sampler_cmd in ['train', 'test']:
            assert os.path.exists(training_options_json), training_options_json
            data_kwargs = train_kwargs['dataset_kwargs']
            if 'cifar' in data_kwargs['path'] and sampler_cmd == 'test':
                dataset_guided = None
            else:
                if sampler_cmd == 'test':
                    if 'ffhq' in data_kwargs['path']:
                        data_kwargs['path'] = data_kwargs['path'].replace('ffhq', 'celebahq')
                        data_kwargs['max_size'] = 30000
                    elif 'celebahq' in data_kwargs['path']:
                        data_kwargs['path'] = data_kwargs['path'].replace('celebahq', 'ffhq')
                        data_kwargs['max_size'] = 70000
                    boxx.cf.kwargs['data'] = data_kwargs['path']
                dist.print0(f'dataset_guided-{sampler_cmd}:', data_kwargs)
                dataset_guided = dnnlib.util.construct_class_by_name(**data_kwargs)
            sampler = ReconstructionDatasetSampler(dataset_guided)
        batch_sampler = BatchedGuidedSampler(sampler)
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=dist.get_rank() != 0):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any((x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling']))
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        if sampler_kwargs.get('discretization', 'ddn') == 'ddn':
            sampler_fn = ddn_sampler
            sampler_kwargs['batch_seeds'] = batch_seeds
            if sampler_cmd:
                sampler_kwargs['sampler'] = batch_sampler
            if markov:
                sampler_kwargs['markov_sampler'] = markov_sampler
        images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        if isinstance(images, dict):
            d, images = (images, images['predict'])
            idx_ks = npa(d['idx_ks']).T
            idx_ks_list.append(idx_ks)
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed - seed % 1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
    torch.distributed.barrier()
    if dist.get_rank() == 0 and len(seeds) >= 9 and (not sampler_cmd):
        example_paths = sorted(glob(outdir + '/**/*.??g', recursive=True))[:100]
        make_vis_img(example_paths, visp)
    if len(seeds) >= 50000:
        ws = dist.get_world_size()
        ts = {rank: -torch.ones((len(seeds) // ws + 1, idx_ks_list[0].shape[-1]), dtype=torch.int32).cuda() for rank in range(ws)}
        for rank, tensor in ts.items():
            if dist.get_rank() == rank:
                idx_ks_ = torch.from_numpy(np.concatenate(idx_ks_list))
                tensor[:len(idx_ks_)] = idx_ks_.to(torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.broadcast(tensor, src=rank)
        if dist.get_rank() == 0:
            ddn_latents = torch.cat(tuple(ts.values()))
            ddn_latents = ddn_latents[ddn_latents[:, 0] >= 0].short().cpu()
            os.makedirs(eval_dir, exist_ok=True)
            ddn_latents_path = os.path.join(eval_dir, sampler_prefix + f'ddn_latents_l{len(ddn_latents[0])}_n{len(ddn_latents)}.pt')
            print('Save DDN latents to:', ddn_latents_path)
            torch.save(ddn_latents, ddn_latents_path)
            if sampler_cmd == 'train':
                torch.save(ddn_latents, os.path.join(eval_dir, 'train_seqs_for_markov.pt'))
        import fid
        if sampler_kwargs.get('fid_ref', None):
            fid_argkws = dict(ref_path=sampler_kwargs['fid_ref'], image_path=outdir, num_expected=min(len(seeds), 50000), seed=0, batch=max_batch_size)
            dist.print0('fid_argkws:', fid_argkws)
            fid = fid.calc_fid(**fid_argkws)
        else:
            dist.print0('fid_argkws:', fid_argkws)
            fid = fid_argkws = {'fid': -1, 'info': 'no fid ref'}
        if dist.get_rank() == 0:
            kimg = ([-1] + boxx.findints(os.path.basename(network_pkl)))[-1]
            os.makedirs(eval_dir, exist_ok=True)
            tmp_tar = '/run/ddn.tar'
            tar_path = os.path.join(eval_dir, sampler_prefix + 'sample-example.tar')
            print('Saving example images tar to:', tar_path)
            example_paths = sorted(glob(outdir + '/**/*.??g', recursive=True))[:100]
            boxx.zipTar(example_paths, tmp_tar)
            copy_file = lambda src, dst: open(dst, 'wb').write(open(src, 'rb').read())
            copy_file(tmp_tar, tar_path)
            make_vis_img(example_paths, os.path.join(eval_dir, sampler_prefix + 'vis.png'))
            boxx.savejson(dict(fid=fid['fid'], path=network_pkl, kimg=kimg, kwargs=boxx.cf.kwargs, fid_argkws=fid_argkws), fid_path)
            boxx.savejson(dict(fid=fid['fid'], path=network_pkl, kimg=kimg), os.path.join(eval_dir, sampler_prefix + 'fid-%.3f' % fid['fid']))
    dist.print0('Done.')
    if boxx.cf.debug:
        sdd = net.model.block_32x32_1.ddo.sdd
        sdd.plot_dist()
    boxx.mg()

def make_vis_img(imgps, visp):
    vis = npa([boxx.imread(pa) for pa in imgps])
    vis_side = int(len(vis) ** 0.5)
    vis = vis[:vis_side ** 2].reshape(vis_side, vis_side, *vis[0].shape)
    boxx.imsave(visp, np.concatenate(np.concatenate(vis, 2), 0))
    print('Save vis to:', visp)

