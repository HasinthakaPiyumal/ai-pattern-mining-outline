# Cluster 5

def main():
    args = parse_args()
    print(args.exp_name)
    if args.dataset == 'vitonhd' and args.vitonhd_dataroot is None:
        raise ValueError('VitonHD dataroot must be provided')
    if args.dataset == 'dresscode' and args.dresscode_dataroot is None:
        raise ValueError('DressCode dataroot must be provided')
    if args.wandb_log:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name, config=vars(args))
    dataset_output_list = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'im_mask', 'pose_map', 'category']
    if args.dense:
        dataset_output_list.append('dense_uv')
    if args.dataset == 'vitonhd':
        dataset_train = VitonHDDataset(phase='train', outputlist=dataset_output_list, dataroot_path=args.vitonhd_dataroot, size=(args.height, args.width))
    elif args.dataset == 'dresscode':
        dataset_train = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='train', outputlist=dataset_output_list, size=(args.height, args.width))
    else:
        raise NotImplementedError('Dataset should be either vitonhd or dresscode')
    dataloader_train = DataLoader(batch_size=args.batch_size, dataset=dataset_train, shuffle=True, num_workers=args.workers)
    if args.dataset == 'vitonhd':
        dataset_test_paired = VitonHDDataset(phase='test', dataroot_path=args.vitonhd_dataroot, outputlist=dataset_output_list, size=(args.height, args.width))
        dataset_test_unpaired = VitonHDDataset(phase='test', order='unpaired', dataroot_path=args.vitonhd_dataroot, outputlist=dataset_output_list, size=(args.height, args.width))
    elif args.dataset == 'dresscode':
        dataset_test_paired = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='test', outputlist=dataset_output_list, size=(args.height, args.width))
        dataset_test_unpaired = DressCodeDataset(phase='test', order='unpaired', dataroot_path=args.dresscode_dataroot, outputlist=dataset_output_list, size=(args.height, args.width))
    else:
        raise NotImplementedError('Dataset should be either vitonhd or dresscode')
    dataloader_test_paired = DataLoader(batch_size=args.batch_size, dataset=dataset_test_paired, shuffle=True, num_workers=args.workers, drop_last=True)
    dataloader_test_unpaired = DataLoader(batch_size=args.batch_size, dataset=dataset_test_unpaired, shuffle=True, num_workers=args.workers, drop_last=True)
    input_nc = 5 if args.dense else 21
    n_layer = 3
    tps = ConvNet_TPS(256, 192, input_nc, n_layer).to(device)
    refinement = UNetVanilla(n_channels=8 if args.dense else 24, n_classes=3, bilinear=True).to(device)
    optimizer_tps = torch.optim.Adam(tps.parameters(), lr=args.lr, betas=(0.5, 0.99))
    optimizer_ref = torch.optim.Adam(list(refinement.parameters()), lr=args.lr, betas=(0.5, 0.99))
    scaler = torch.cuda.amp.GradScaler()
    criterion_l1 = nn.L1Loss()
    if args.vgg_weight > 0:
        criterion_vgg = VGGLoss().to(device)
    else:
        criterion_vgg = None
    start_epoch = 0
    if os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name, f'checkpoint_last.pth')):
        print('Loading full checkpoint')
        state_dict = torch.load(os.path.join(args.checkpoints_dir, args.exp_name, f'checkpoint_last.pth'))
        tps.load_state_dict(state_dict['tps'])
        refinement.load_state_dict(state_dict['refinement'])
        optimizer_tps.load_state_dict(state_dict['optimizer_tps'])
        optimizer_ref.load_state_dict(state_dict['optimizer_ref'])
        start_epoch = state_dict['epoch']
        if args.only_extraction:
            print('Extracting warped cloth images...')
            extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
            extraction_dataloader_paired = DataLoader(batch_size=args.batch_size, dataset=extraction_dataset_paired, shuffle=False, num_workers=args.workers, drop_last=False)
            if args.save_path:
                warped_cloth_root = args.save_path
            else:
                warped_cloth_root = PROJECT_ROOT / 'data'
            save_name_paired = warped_cloth_root / 'warped_cloths' / args.dataset
            extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, args.height, args.width)
            extraction_dataset = dataset_test_unpaired
            extraction_dataloader_paired = DataLoader(batch_size=args.batch_size, dataset=extraction_dataset, shuffle=False, num_workers=args.workers)
            save_name_unpaired = warped_cloth_root / 'warped_cloths_unpaired' / args.dataset
            extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, args.height, args.width)
            exit()
    if args.only_extraction and (not os.path.exists(os.path.join(args.checkpoints_dir, args.exp_name, f'checkpoint_last.pth'))):
        print('No checkpoint found, before extracting warped cloth images, please train the model first.')
        exit()
    dataset_train.height = 256
    dataset_train.width = 192
    for e in range(start_epoch, args.epochs_tps):
        print(f'Epoch {e}/{args.epochs_tps}')
        print('train')
        train_loss, train_l1_loss, train_const_loss, visual = training_loop_tps(dataloader_train, tps, optimizer_tps, criterion_l1, scaler, args.const_weight)
        print('paired test')
        running_loss, vgg_running_loss, visual = compute_metric(dataloader_test_paired, tps, criterion_l1, criterion_vgg, refinement=None, height=args.height, width=args.width)
        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        print('unpaired test')
        running_loss_unpaired, vgg_running_loss_unpaired, visual = compute_metric(dataloader_test_unpaired, tps, criterion_l1, criterion_vgg, refinement=None, height=args.height, width=args.width)
        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        if args.wandb_log:
            wandb.log({'train/loss': train_loss, 'train/l1_loss': train_l1_loss, 'train/const_loss': train_const_loss, 'train/vgg_loss': 0, 'eval/eval_loss_paired': running_loss, 'eval/eval_vgg_loss_paired': vgg_running_loss, 'eval/eval_loss_unpaired': running_loss_unpaired, 'eval/eval_vgg_loss_unpaired': vgg_running_loss_unpaired, 'images_paired': wandb.Image(imgs), 'images_unpaired': wandb.Image(imgs_unpaired)})
        os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)
        torch.save({'epoch': e + 1, 'tps': tps.state_dict(), 'refinement': refinement.state_dict(), 'optimizer_tps': optimizer_tps.state_dict(), 'optimizer_ref': optimizer_ref.state_dict()}, os.path.join(args.checkpoints_dir, args.exp_name, f'checkpoint_last.pth'))
    scaler = torch.cuda.amp.GradScaler()
    dataset_train.height = args.height
    dataset_train.width = args.width
    for e in range(max(start_epoch, args.epochs_tps), max(start_epoch, args.epochs_tps) + args.epochs_refinement):
        print(f'Epoch {e}/{max(start_epoch, args.epochs_tps) + args.epochs_refinement}')
        train_loss, train_l1_loss, train_vgg_loss, visual = training_loop_refinement(dataloader_train, tps, refinement, optimizer_ref, criterion_l1, criterion_vgg, args.l1_weight, args.vgg_weight, scaler, args.height, args.width)
        running_loss, vgg_running_loss, visual = compute_metric(dataloader_test_paired, tps, criterion_l1, criterion_vgg, refinement=refinement, height=args.height, width=args.width)
        imgs = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        running_loss_unpaired, vgg_running_loss_unpaired, visual = compute_metric(dataloader_test_unpaired, tps, criterion_l1, criterion_vgg, refinement=refinement, height=args.height, width=args.width)
        imgs_unpaired = torchvision.utils.make_grid(torch.cat(visual[0]), nrow=len(visual[0][0]), padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        if args.wandb_log:
            wandb.log({'train/loss': train_loss, 'train/l1_loss': train_l1_loss, 'train/const_loss': 0, 'train/vgg_loss': train_vgg_loss, 'eval/eval_loss_paired': running_loss, 'eval/eval_vgg_loss_paired': vgg_running_loss, 'eval/eval_loss_unpaired': running_loss_unpaired, 'eval/eval_vgg_loss_unpaired': vgg_running_loss_unpaired, 'images_paired': wandb.Image(imgs), 'images_unpaired': wandb.Image(imgs_unpaired)})
        os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)
        torch.save({'epoch': e + 1, 'tps': tps.state_dict(), 'refinement': refinement.state_dict(), 'optimizer_tps': optimizer_tps.state_dict(), 'optimizer_ref': optimizer_ref.state_dict()}, os.path.join(args.checkpoints_dir, args.exp_name, f'checkpoint_last.pth'))
    print('Extracting warped cloth images...')
    extraction_dataset_paired = torch.utils.data.ConcatDataset([dataset_test_paired, dataset_train])
    extraction_dataloader_paired = DataLoader(batch_size=args.batch_size, dataset=extraction_dataset_paired, shuffle=False, num_workers=args.workers, drop_last=False)
    if args.save_path:
        warped_cloth_root = args.save_path
    else:
        warped_cloth_root = PROJECT_ROOT / 'data'
    save_name_paired = warped_cloth_root / 'warped_cloths' / args.dataset
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_paired, args.height, args.width)
    extraction_dataset = dataset_test_unpaired
    extraction_dataloader_paired = DataLoader(batch_size=args.batch_size, dataset=extraction_dataset, shuffle=False, num_workers=args.workers)
    save_name_unpaired = warped_cloth_root / 'warped_cloths_unpaired' / args.dataset
    extract_images(extraction_dataloader_paired, tps, refinement, save_name_unpaired, args.height, args.width)

@torch.no_grad()
def extract_images(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla, save_path: str, height: int=512, width: int=384) -> None:
    """
    Extracts the images using the trained networks and saves them to the save_path
    """
    tps.eval()
    refinement.eval()
    for step, inputs in enumerate(tqdm(dataloader)):
        c_name = inputs['c_name']
        im_name = inputs['im_name']
        cloth = inputs['cloth'].to(device)
        category = inputs.get('category')
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(height, width), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)
        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth)
        warped_cloth = (warped_cloth + 1) / 2
        warped_cloth = warped_cloth.clamp(0, 1)
        for cname, iname, warpclo, cat in zip(c_name, im_name, warped_cloth, category):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            save_image(warpclo, os.path.join(save_path, cat, iname.replace('.jpg', '') + '_' + cname), quality=95)

def training_loop_tps(dataloader: DataLoader, tps: ConvNet_TPS, optimizer_tps: torch.optim.Optimizer, criterion_l1: nn.L1Loss, scaler: torch.cuda.amp.GradScaler, const_weight: float) -> tuple[float, float, float, list[list]]:
    """
    Training loop for the TPS network. Note that the TPS is trained on a low resolution image for sake of performance.
    """
    tps.train()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_const_loss = 0.0
    for step, inputs in enumerate(tqdm(dataloader)):
        low_cloth = inputs['cloth'].to(device, non_blocking=True)
        low_image = inputs['image'].to(device, non_blocking=True)
        low_im_cloth = inputs['im_cloth'].to(device, non_blocking=True)
        low_im_mask = inputs['im_mask'].to(device, non_blocking=True)
        low_pose_map = inputs.get('dense_uv')
        if low_pose_map is None:
            low_pose_map = inputs['pose_map']
        low_pose_map = low_pose_map.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
            low_warped_cloth = F.grid_sample(low_cloth, low_grid, padding_mode='border')
            l1_loss = criterion_l1(low_warped_cloth, low_im_cloth)
            const_loss = torch.mean(rx + ry + cx + cy + rg + cg)
            loss = l1_loss + const_loss * const_weight
        optimizer_tps.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_tps)
        scaler.update()
        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_const_loss += const_loss.item()
    visual = [[low_image, low_cloth, low_im_cloth, low_warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    const_loss = running_const_loss / (step + 1)
    return (loss, l1_loss, const_loss, visual)

@torch.no_grad()
def compute_metric(dataloader: DataLoader, tps: ConvNet_TPS, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss, refinement: UNetVanilla=None, height: int=512, width: int=384) -> tuple[float, float, list[list]]:
    """
    Perform inference on the given dataloader and compute the L1 and VGG loss between the warped cloth and the
    ground truth image.
    """
    tps.eval()
    if refinement:
        refinement.eval()
    running_loss = 0.0
    vgg_running_loss = 0
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(height, width), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)
        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')
        if refinement:
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)
        loss = criterion_l1(warped_cloth, im_cloth)
        running_loss += loss.item()
        if criterion_vgg:
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)
            vgg_running_loss += vgg_loss.item()
    visual = [[image, cloth, im_cloth, warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    vgg_loss = vgg_running_loss / (step + 1)
    return (loss, vgg_loss, visual)

def training_loop_refinement(dataloader: DataLoader, tps: ConvNet_TPS, refinement: UNetVanilla, optimizer_ref: torch.optim.Optimizer, criterion_l1: nn.L1Loss, criterion_vgg: VGGLoss, l1_weight: float, vgg_weight: float, scaler: torch.cuda.amp.GradScaler, height=512, width=384) -> tuple[float, float, float, list[list]]:
    """
    Training loop for the refinement network. Note that the refinement network is trained on a high resolution image
    """
    tps.eval()
    refinement.train()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_vgg_loss = 0.0
    for step, inputs in enumerate(tqdm(dataloader)):
        cloth = inputs['cloth'].to(device)
        image = inputs['image'].to(device)
        im_cloth = inputs['im_cloth'].to(device)
        im_mask = inputs['im_mask'].to(device)
        pose_map = inputs.get('dense_uv')
        if pose_map is None:
            pose_map = inputs['pose_map']
        pose_map = pose_map.to(device)
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        with torch.cuda.amp.autocast():
            agnostic = torch.cat([low_im_mask, low_pose_map], 1)
            low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)
            low_warped_cloth = F.grid_sample(cloth, low_grid, padding_mode='border')
            highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(height, width), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)
            warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')
            warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
            warped_cloth = refinement(warped_cloth)
            l1_loss = criterion_l1(warped_cloth, im_cloth)
            vgg_loss = criterion_vgg(warped_cloth, im_cloth)
            loss = l1_loss * l1_weight + vgg_loss * vgg_weight
        optimizer_ref.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_ref)
        scaler.update()
        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_vgg_loss += vgg_loss.item()
    visual = [[image, cloth, im_cloth, low_warped_cloth.clamp(-1, 1)]]
    loss = running_loss / (step + 1)
    l1_loss = running_l1_loss / (step + 1)
    vgg_loss = running_vgg_loss / (step + 1)
    return (loss, l1_loss, vgg_loss, visual)

