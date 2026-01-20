# Cluster 6

@torch.no_grad()
def main():
    args = parse_args()
    if args.dataset == 'vitonhd' and args.vitonhd_dataroot is None:
        raise ValueError('VitonHD dataroot must be provided')
    if args.dataset == 'dresscode' and args.dresscode_dataroot is None:
        raise ValueError('DressCode dataroot must be provided')
    accelerator = Accelerator()
    device = accelerator.device
    if args.pretrained_model_name_or_path == 'runwayml/stable-diffusion-inpainting':
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
    elif args.pretrained_model_name_or_path == 'stabilityai/stable-diffusion-2-inpainting':
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    else:
        raise ValueError(f'Unknown pretrained model name or path: {args.pretrained_model_name_or_path}')
    vision_encoder.requires_grad_(False)
    vision_encoder = vision_encoder.to(device)
    outputlist = ['cloth', 'c_name']
    if args.dataset == 'dresscode':
        train_dataset = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='train', order='paired', radius=5, category=['dresses', 'upper_body', 'lower_body'], size=(512, 384), outputlist=tuple(outputlist))
        test_dataset = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='test', order='paired', radius=5, category=['dresses', 'upper_body', 'lower_body'], size=(512, 384), outputlist=tuple(outputlist))
    elif args.dataset == 'vitonhd':
        train_dataset = VitonHDDataset(dataroot_path=args.vitonhd_dataroot, phase='train', order='paired', radius=5, size=(512, 384), outputlist=tuple(outputlist))
        test_dataset = VitonHDDataset(dataroot_path=args.vitonhd_dataroot, phase='test', order='paired', radius=5, size=(512, 384), outputlist=tuple(outputlist))
    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    save_cloth_features(args.dataset, processor, train_loader, vision_encoder, 'train')
    save_cloth_features(args.dataset, processor, test_loader, vision_encoder, 'test')

