# Cluster 7

def compute_metrics(gen_folder: str, test_order: str, dataset: str, category: str, metrics2compute: List[str], dresscode_dataroot: str, vitonhd_dataroot: str, generated_size: Tuple[int, int]=(512, 384), batch_size: int=32, workers: int=8) -> Dict[str, float]:
    """
    Computes the metrics for the generated images in gen_folder
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert test_order in ['paired', 'unpaired']
    assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
    assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'
    if dataset == 'dresscode':
        gt_folder = dresscode_dataroot
    elif dataset == 'vitonhd':
        gt_folder = vitonhd_dataroot
    else:
        raise ValueError('Unsupported dataset')
    for m in metrics2compute:
        assert m in ['all', 'ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score'], 'Unsupported metric'
    if metrics2compute == ['all']:
        metrics2compute = ['ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score']
    if category == 'all':
        if 'fid_score' in metrics2compute or 'all' in metrics2compute:
            if not fid.test_stats_exists(f'{dataset}_all', mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(gen_folder, dataset_name=f'{dataset}_all', mode='clean', dataset_split='custom', verbose=True, use_dataparallel=False)
        if 'kid_score' in metrics2compute or 'all' in metrics2compute:
            if not fid.test_stats_exists(f'{dataset}_all', mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(gen_folder, dataset_name=f'{dataset}_all', mode='clean', dataset_split='custom', verbose=True, use_dataparallel=False)
    else:
        if 'fid_score' in metrics2compute or 'all' in metrics2compute:
            if not fid.test_stats_exists(f'{dataset}_{category}', mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            fid_score = fid.compute_fid(os.path.join(gen_folder, category), dataset_name=f'{dataset}_{category}', mode='clean', verbose=True, dataset_split='custom', use_dataparallel=False)
        if 'kid_score' in metrics2compute or 'all' in metrics2compute:
            if not fid.test_stats_exists(f'{dataset}_{category}', mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)
            kid_score = fid.compute_kid(os.path.join(gen_folder, category), dataset_name=f'{dataset}_{category}', mode='clean', verbose=True, dataset_split='custom', use_dataparallel=False)
    trans = transforms.Compose([transforms.Resize(generated_size), transforms.ToTensor()])
    gen_dataset = GenTestDataset(gen_folder, category, transform=trans)
    gt_dataset = GTTestDataset(gt_folder, dataset, category, trans)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    if 'is_score' in metrics2compute or 'all' in metrics2compute:
        model_is = InceptionScore(normalize=True).to(device)
    if 'ssim_score' in metrics2compute or 'all' in metrics2compute:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if 'lpips_score' in metrics2compute or 'all' in metrics2compute:
        lpips = LearnedPerceptualImagePatchSimilarity(net='alex', normalize=True).to(device)
    for idx, (gen_batch, gt_batch) in tqdm(enumerate(zip(gen_loader, gt_loader)), total=len(gt_loader)):
        gen_images, gen_names = gen_batch
        gt_images, gt_names = gt_batch
        assert gen_names == gt_names
        gen_images = gen_images.to(device)
        gt_images = gt_images.to(device)
        if 'is_score' in metrics2compute or 'all' in metrics2compute:
            model_is.update(gen_images)
        if 'ssim_score' in metrics2compute or 'all' in metrics2compute:
            ssim.update(gen_images, gt_images)
        if 'lpips_score' in metrics2compute or 'all' in metrics2compute:
            lpips.update(gen_images, gt_images)
    if 'is_score' in metrics2compute or 'all' in metrics2compute:
        is_score, is_std = model_is.compute()
    if 'ssim_score' in metrics2compute or 'all' in metrics2compute:
        ssim_score = ssim.compute()
    if 'lpips_score' in metrics2compute or 'all' in metrics2compute:
        lpips_score = lpips.compute()
    results = {}
    for m in metrics2compute:
        if torch.is_tensor(locals()[m]):
            results[m] = locals()[m].item()
        else:
            results[m] = locals()[m]
    return results

def make_custom_stats(dresscode_dataroot: str, vitonhd_dataroot: str):
    if dresscode_dataroot is not None:
        dresscode_filesplit = os.path.join(dresscode_dataroot, f'test_pairs_paired.txt')
        with open(dresscode_filesplit, 'r') as f:
            lines = f.read().splitlines()
        for category in ['lower_body', 'upper_body', 'dresses']:
            if not fid.test_stats_exists(f'dresscode_{category}', mode='clean'):
                paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines if os.path.exists(os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]))]
                tmp_folder = f'/tmp/dresscode/{category}'
                os.makedirs(tmp_folder, exist_ok=True)
                for path in tqdm(paths):
                    shutil.copy(path, tmp_folder)
                fid.make_custom_stats(f'dresscode_{category}', tmp_folder, mode='clean', verbose=True)
        if not fid.test_stats_exists(f'dresscode_all', mode='clean'):
            paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines for category in ['lower_body', 'upper_body', 'dresses'] if os.path.exists(os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]))]
            tmp_folder = f'/tmp/dresscode/all'
            os.makedirs(tmp_folder, exist_ok=True)
            for path in tqdm(paths):
                shutil.copy(path, tmp_folder)
            fid.make_custom_stats(f'dresscode_all', tmp_folder, mode='clean', verbose=True)
    if vitonhd_dataroot is not None:
        if not fid.test_stats_exists(f'vitonhd_all', mode='clean'):
            fid.make_custom_stats(f'vitonhd_all', os.path.join(vitonhd_dataroot, 'test', 'image'), mode='clean', verbose=True)
        if not fid.test_stats_exists(f'vitonhd_upper_body', mode='clean'):
            fid.make_custom_stats(f'vitonhd_upper_body', os.path.join(vitonhd_dataroot, 'test', 'image'), mode='clean', verbose=True)

