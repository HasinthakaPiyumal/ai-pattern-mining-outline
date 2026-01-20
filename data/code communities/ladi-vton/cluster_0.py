# Cluster 0

def inversion_adapter(dataset: Literal['dresscode', 'vitonhd']):
    config = AutoConfig.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    text_encoder_config = UNet2DConditionModel.load_config('stabilityai/stable-diffusion-2-inpainting', subfolder='text_encoder')
    inversion_adapter = InversionAdapter(input_dim=config.vision_config.hidden_size, hidden_dim=config.vision_config.hidden_size * 4, output_dim=text_encoder_config['hidden_size'] * 16, num_encoder_layers=1, config=config.vision_config)
    checkpoint_url = f'https://github.com/miccunifi/ladi-vton/releases/download/weights/inversion_adapter_{dataset}.pth'
    inversion_adapter.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return inversion_adapter

@torch.inference_mode()
def main():
    args = parse_args()
    if args.dataset == 'vitonhd' and args.vitonhd_dataroot is None:
        raise ValueError('VitonHD dataroot must be provided')
    if args.dataset == 'dresscode' and args.dresscode_dataroot is None:
        raise ValueError('DressCode dataroot must be provided')
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed)
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    val_scheduler.set_timesteps(50, device=device)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    processor = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet', dataset=args.dataset)
    emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=args.dataset)
    inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter', dataset=args.dataset)
    tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', dataset=args.dataset)
    int_layers = [1, 2, 3, 4, 5]
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError('xformers is not available. Make sure it is installed correctly')
    if args.category != 'all':
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']
    outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']
    if args.dataset == 'dresscode':
        test_dataset = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='test', order=args.test_order, radius=5, outputlist=outputlist, category=category, size=(512, 384))
    elif args.dataset == 'vitonhd':
        test_dataset = VitonHDDataset(dataroot_path=args.vitonhd_dataroot, phase='test', order=args.test_order, radius=5, outputlist=outputlist, size=(512, 384))
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    emasc.to(device, dtype=weight_dtype)
    inversion_adapter.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    tps.to(device, dtype=torch.float32)
    refinement.to(device, dtype=torch.float32)
    vision_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()
    vae.eval()
    emasc.eval()
    inversion_adapter.eval()
    unet.eval()
    tps.eval()
    refinement.eval()
    vision_encoder.eval()
    val_pipe = StableDiffusionTryOnePipeline(text_encoder=text_encoder, vae=vae, tokenizer=tokenizer, unet=unet, scheduler=val_scheduler, emasc=emasc, emasc_int_layers=int_layers).to(device)
    test_dataloader = accelerator.prepare(test_dataloader)
    save_dir = os.path.join(args.output_dir, args.test_order)
    os.makedirs(save_dir, exist_ok=True)
    generator = torch.Generator('cuda').manual_seed(args.seed)
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get('image').to(weight_dtype)
        mask_img = batch.get('inpaint_mask').to(weight_dtype)
        if mask_img is not None:
            mask_img = mask_img.to(weight_dtype)
        pose_map = batch.get('pose_map').to(weight_dtype)
        category = batch.get('category')
        cloth = batch.get('cloth').to(weight_dtype)
        im_mask = batch.get('im_mask').to(weight_dtype)
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192), torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth.to(torch.float32), agnostic.to(torch.float32))
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2), size=(512, 384), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).permute(0, 2, 3, 1)
        warped_cloth = F.grid_sample(cloth.to(torch.float32), highres_grid.to(torch.float32), padding_mode='border')
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth.to(torch.float32))
        warped_cloth = warped_cloth.clamp(-1, 1)
        warped_cloth = warped_cloth.to(weight_dtype)
        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
        processed_images = processor(images=input_image, return_tensors='pt')
        clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device, dtype=weight_dtype)).last_hidden_state
        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], args.num_vstar, -1))
        category_text = {'dresses': 'a dress', 'upper_body': 'an upper body garment', 'lower_body': 'a lower body garment'}
        text = [f'a photo of a model wearing {category_text[category]} {' $ ' * args.num_vstar}' for category in batch['category']]
        tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
        tokenized_text = tokenized_text.to(word_embeddings.device)
        encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text, word_embeddings, args.num_vstar).last_hidden_state
        generated_images = val_pipe(image=model_img, mask_image=mask_img, pose_map=pose_map, warped_cloth=warped_cloth, prompt_embeds=encoder_hidden_states, height=512, width=384, guidance_scale=args.guidance_scale, num_images_per_prompt=1, generator=generator, cloth_input_type='warped', num_inference_steps=args.num_inference_steps).images
        for gen_image, cat, name in zip(generated_images, category, batch['im_name']):
            if not os.path.exists(os.path.join(save_dir, cat)):
                os.makedirs(os.path.join(save_dir, cat))
            if args.use_png:
                name = name.replace('.jpg', '.png')
                gen_image.save(os.path.join(save_dir, cat, name))
            else:
                gen_image.save(os.path.join(save_dir, cat, name), quality=95)
    del val_pipe
    del text_encoder
    del vae
    del emasc
    del unet
    del tps
    del refinement
    del vision_encoder
    torch.cuda.empty_cache()
    if args.compute_metrics:
        metrics = compute_metrics(save_dir, args.test_order, args.dataset, args.category, ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)
        with open(os.path.join(save_dir, f'metrics_{args.test_order}_{args.category}.json'), 'w+') as f:
            json.dump(metrics, f, indent=4)

def encode_text_word_embedding(text_encoder: CLIPTextModel, input_ids: torch.tensor, word_embeddings: torch.tensor, num_vstar: int=1) -> BaseModelOutputWithPooling:
    """
    Encode text by replacing the '$' with the PTEs extracted with the inversion adapter.
    Heavily based on hugginface implementation of CLIP.
    """
    existing_indexes = (input_ids == 259).nonzero(as_tuple=True)[0]
    existing_indexes = existing_indexes.unique()
    if len(existing_indexes) > 0:
        _, counts = torch.unique((input_ids == 259).nonzero(as_tuple=True)[0], return_counts=True)
        cum_sum = torch.cat((torch.zeros(1, device=input_ids.device).int(), torch.cumsum(counts, dim=0)[:-1]))
        first_vstar_indexes = (input_ids == 259).nonzero()[cum_sum][:, 1]
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_vstar)])
        word_embeddings = word_embeddings.to(input_ids.device)
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    seq_length = input_ids.shape[-1]
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :seq_length]
    input_embeds = text_encoder.text_model.embeddings.token_embedding(input_ids)
    if len(existing_indexes) > 0:
        assert word_embeddings.shape[0] == input_embeds.shape[0]
        if len(word_embeddings.shape) == 2:
            word_embeddings = word_embeddings.unsqueeze(1)
        input_embeds[torch.arange(input_embeds.shape[0]).repeat_interleave(num_vstar).reshape(input_embeds.shape[0], num_vstar)[existing_indexes.cpu()], rep_idx.T] = word_embeddings.to(input_embeds.dtype)[existing_indexes]
    position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
    hidden_states = input_embeds + position_embeddings
    bsz, seq_len = input_shape
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(hidden_states.device)
    encoder_outputs = text_encoder.text_model.encoder(inputs_embeds=hidden_states, attention_mask=None, causal_attention_mask=causal_attention_mask, output_attentions=None, output_hidden_states=None, return_dict=None)
    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)]
    return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

@torch.no_grad()
def generate_images_from_tryon_pipe(pipe: StableDiffusionTryOnePipeline, inversion_adapter: InversionAdapter, test_dataloader: torch.utils.data.DataLoader, output_dir: str, order: str, save_name: str, text_usage: str, vision_encoder: CLIPVisionModelWithProjection, processor: CLIPProcessor, cloth_input_type: str, cloth_cond_rate: int=1, num_vstar: int=1, seed: int=1234, num_inference_steps: int=50, guidance_scale: int=7.5, use_png: bool=False):
    save_path = os.path.join(output_dir, f'{save_name}_{order}')
    os.makedirs(save_path, exist_ok=True)
    generator = torch.Generator('cuda').manual_seed(seed)
    num_samples = 1
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch.get('image')
        mask_img = batch.get('inpaint_mask')
        if mask_img is not None:
            mask_img = mask_img.type(torch.float32)
        pose_map = batch.get('pose_map')
        warped_cloth = batch.get('warped_cloth')
        category = batch.get('category')
        cloth = batch.get('cloth')
        if text_usage == 'noun_chunks':
            prompts = batch['captions']
        elif text_usage == 'none':
            prompts = [''] * len(batch['captions'])
        elif text_usage == 'inversion_adapter':
            category_text = {'dresses': 'a dress', 'upper_body': 'an upper body garment', 'lower_body': 'a lower body garment'}
            text = [f'a photo of a model wearing {category_text[category]} {' $ ' * num_vstar}' for category in batch['category']]
            clip_cloth_features = batch.get('clip_cloth_features')
            if clip_cloth_features is None:
                with torch.no_grad():
                    input_image = torchvision.transforms.functional.resize((batch['cloth'] + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
                    processed_images = processor(images=input_image, return_tensors='pt')
                    clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device)).last_hidden_state
            word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
            word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))
            tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
            tokenized_text = tokenized_text.to(word_embeddings.device)
            encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text, word_embeddings, num_vstar).last_hidden_state
        else:
            raise ValueError(f'Unknown text usage {text_usage}')
        if text_usage == 'inversion_adapter':
            generated_images = pipe(image=model_img, mask_image=mask_img, pose_map=pose_map, warped_cloth=warped_cloth, prompt_embeds=encoder_hidden_states, height=512, width=384, guidance_scale=guidance_scale, num_images_per_prompt=num_samples, generator=generator, cloth_input_type=cloth_input_type, cloth_cond_rate=cloth_cond_rate, num_inference_steps=num_inference_steps).images
        else:
            generated_images = pipe(prompt=prompts, image=model_img, mask_image=mask_img, pose_map=pose_map, warped_cloth=warped_cloth, height=512, width=384, guidance_scale=guidance_scale, num_images_per_prompt=num_samples, generator=generator, cloth_input_type=cloth_input_type, cloth_cond_rate=cloth_cond_rate, num_inference_steps=num_inference_steps).images
        for gen_image, cat, name in zip(generated_images, category, batch['im_name']):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            if use_png:
                name = name.replace('.jpg', '.png')
                gen_image.save(os.path.join(save_path, cat, name))
            else:
                gen_image.save(os.path.join(save_path, cat, name), quality=95)

def generate_images_inversion_adapter(pipe: StableDiffusionInpaintPipeline, inversion_adapter: InversionAdapter, vision_encoder: CLIPVisionModelWithProjection, processor: CLIPProcessor, test_dataloader: torch.utils.data.DataLoader, output_dir, order: str, save_name: str, num_vstar=1, seed=1234, num_inference_steps=50, guidance_scale=7.5, use_png=False) -> None:
    """
    Extract and save images using the SD inpainting pipeline using the PTEs from the inversion adapter.
    """
    save_path = os.path.join(output_dir, f'{save_name}_{order}')
    os.makedirs(save_path, exist_ok=True)
    generator = torch.Generator('cuda').manual_seed(seed)
    num_samples = 1
    for idx, batch in enumerate(tqdm(test_dataloader)):
        model_img = batch['image']
        mask_img = batch['inpaint_mask']
        mask_img = mask_img.type(torch.float32)
        category = batch['category']
        cloth = batch.get('cloth')
        clip_cloth_features = batch.get('clip_cloth_features')
        if clip_cloth_features is None:
            input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224), antialias=True).clamp(0, 1)
            processed_images = processor(images=input_image, return_tensors='pt')
            clip_cloth_features = vision_encoder(processed_images.pixel_values.to(model_img.device)).last_hidden_state
        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], num_vstar, -1))
        category_text = {'dresses': 'a dress', 'upper_body': 'an upper body garment', 'lower_body': 'a lower body garment'}
        text = [f'a photo of a model wearing {category_text[category]} {' $ ' * num_vstar}' for category in batch['category']]
        tokenized_text = pipe.tokenizer(text, max_length=pipe.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
        tokenized_text = tokenized_text.to(model_img.device)
        encoder_hidden_states = encode_text_word_embedding(pipe.text_encoder, tokenized_text, word_embeddings, num_vstar=num_vstar).last_hidden_state
        generated_images = pipe(image=model_img, mask_image=mask_img, prompt_embeds=encoder_hidden_states, height=512, width=384, guidance_scale=guidance_scale, num_images_per_prompt=num_samples, generator=generator, num_inference_steps=num_inference_steps).images
        for gen_image, cat, name in zip(generated_images, category, batch['im_name']):
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            if use_png:
                name = name.replace('.jpg', '.png')
                gen_image.save(os.path.join(save_path, cat, name))
            else:
                gen_image.save(os.path.join(save_path, cat, name), quality=95)

