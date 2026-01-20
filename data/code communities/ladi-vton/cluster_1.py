# Cluster 1

def emasc(dataset: Literal['dresscode', 'vitonhd']):
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    emasc = EMASC(in_feature_channels, out_feature_channels, kernel_size=3, padding=1, stride=1, type='nonlinear')
    checkpoint_url = f'https://github.com/miccunifi/ladi-vton/releases/download/weights/emasc_{dataset}.pth'
    emasc.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu'))
    return emasc

def main():
    args = parse_args()
    if args.dataset == 'vitonhd' and args.vitonhd_dataroot is None:
        raise ValueError('VitonHD dataroot must be provided')
    if args.dataset == 'dresscode' and args.dresscode_dataroot is None:
        raise ValueError('DressCode dataroot must be provided')
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=args.report_to)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    vae.eval()
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]
    emasc = EMASC(in_feature_channels, out_feature_channels, kernel_size=args.emasc_kernel, padding=args.emasc_padding, stride=1, type=args.emasc_type)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    optimizer = torch.optim.AdamW(emasc.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    if args.dataset == 'dresscode':
        train_dataset = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='train', order='paired', radius=5, category=['dresses', 'upper_body', 'lower_body'], size=(512, 384), outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name'))
        test_dataset = DressCodeDataset(dataroot_path=args.dresscode_dataroot, phase='test', order=args.test_order, radius=5, category=['dresses', 'upper_body', 'lower_body'], size=(512, 384), outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name'))
    elif args.dataset == 'vitonhd':
        train_dataset = VitonHDDataset(dataroot_path=args.vitonhd_dataroot, phase='train', order='paired', radius=5, size=(512, 384), outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name'))
        test_dataset = VitonHDDataset(dataroot_path=args.vitonhd_dataroot, phase='test', order=args.test_order, radius=5, size=(512, 384), outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name'))
    else:
        raise NotImplementedError('Dataset not implemented')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, num_workers=args.num_workers_test)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)
    if args.vgg_weight > 0:
        criterion_vgg = VGGLoss()
    else:
        criterion_vgg = None
    emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg = accelerator.prepare(emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers('LaDI_VTON_EMASC', config=vars(args), init_kwargs={'wandb': {'name': os.path.basename(args.output_dir)}})
        if args.report_to == 'wandb':
            wandb_tracker = accelerator.get_tracker('wandb')
            wandb_tracker.name = os.path.basename(args.output_dir)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint != 'latest':
                path = os.path.basename(os.path.join('checkpoint', args.resume_from_checkpoint))
            else:
                dirs = os.listdir(os.path.join(args.output_dir, 'checkpoint'))
                dirs = [d for d in dirs if d.startswith('checkpoint')]
                dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
                path = dirs[-1]
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(args.output_dir, 'checkpoint', path))
            global_step = int(path.split('-')[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        except Exception as e:
            print('Failed to load checkpoint, training from scratch:')
            print(e)
            resume_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Steps')
    for epoch in range(first_epoch, args.num_train_epochs):
        emasc.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and (step < resume_step):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(emasc):
                with torch.no_grad():
                    posterior_im, _ = vae.encode(batch['image'])
                    _, intermediate_features = vae.encode(batch['im_mask'])
                    intermediate_features = [intermediate_features[i] for i in int_layers]
                processed_intermediate_features = emasc(intermediate_features)
                processed_intermediate_features = mask_features(processed_intermediate_features, batch['inpaint_mask'])
                latents = posterior_im.latent_dist.sample()
                reconstructed_image = vae.decode(z=latents, intermediate_features=processed_intermediate_features, int_layers=int_layers).sample
                with accelerator.autocast():
                    loss = F.l1_loss(reconstructed_image, batch['image'], reduction='mean')
                    if criterion_vgg:
                        loss += args.vgg_weight * criterion_vgg(reconstructed_image, batch['image'])
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(emasc.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    emasc.eval()
                    if accelerator.is_main_process:
                        os.makedirs(os.path.join(args.output_dir, 'checkpoint'), exist_ok=True)
                        accelerator_state_path = os.path.join(args.output_dir, 'checkpoint', f'checkpoint-{global_step}')
                        accelerator.save_state(accelerator_state_path)
                        unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                extract_save_vae_images(vae, unwrapped_emasc, test_dataloader, int_layers, args.output_dir, args.test_order, save_name=f'imgs_step_{global_step}', emasc_type=args.emasc_type)
                            metrics = compute_metrics(os.path.join(args.output_dir, f'imgs_step_{global_step}_{args.test_order}'), args.test_order, args.dataset, 'all', ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)
                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)
                            dirs = os.listdir(os.path.join(args.output_dir, 'checkpoint'))
                            dirs = [d for d in dirs if d.startswith('checkpoint')]
                            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
                            try:
                                path = dirs[-2]
                                shutil.rmtree(os.path.join(args.output_dir, 'checkpoint', path), ignore_errors=True)
                            except:
                                print('No checkpoint to delete')
                            emasc_path = os.path.join(args.output_dir, f'emasc_{global_step}.pth')
                            accelerator.save(unwrapped_emasc.state_dict(), emasc_path)
                            del unwrapped_emasc
                        emasc.train()
            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    accelerator.end_training()

@torch.inference_mode()
def extract_save_vae_images(vae: AutoencoderKL, emasc: EMASC, test_dataloader: torch.utils.data.DataLoader, int_layers: List[int], output_dir: str, order: str, save_name: str, emasc_type: str) -> None:
    """
    Extract and save image using only VAE or VAE + EMASC
    """
    save_path = os.path.join(output_dir, f'{save_name}_{order}')
    os.makedirs(save_path, exist_ok=True)
    for idx, batch in enumerate(tqdm(test_dataloader)):
        category = batch['category']
        if emasc_type != 'none':
            posterior_im, _ = vae.encode(batch['image'])
            _, intermediate_features = vae.encode(batch['im_mask'])
            intermediate_features = [intermediate_features[i] for i in int_layers]
            processed_intermediate_features = emasc(intermediate_features)
            processed_intermediate_features = mask_features(processed_intermediate_features, batch['inpaint_mask'])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents, processed_intermediate_features, int_layers).sample
        else:
            posterior_im = vae.encode(batch['image'])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents).sample
        for gen_image, cat, name in zip(generated_images, category, batch['im_name']):
            gen_image = (gen_image + 1) / 2
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            torchvision.utils.save_image(gen_image, os.path.join(save_path, cat, name), quality=95)

def mask_features(features: list, mask: torch.Tensor):
    """
    Mask features with the given mask.
    """
    for i, feature in enumerate(features):
        mask = torch.nn.functional.interpolate(mask, size=feature.shape[-2:])
        features[i] = feature * (1 - mask)
    return features

class StableDiffusionTryOnePipeline(DiffusionPipeline):
    """
    Pipeline for text and posemap -guided image inpainting using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ['safety_checker']

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler], safety_checker=None, feature_extractor=None, requires_safety_checker: bool=False, emasc=None, emasc_int_layers=None):
        super().__init__()
        self.emasc = emasc
        self.emasc_int_layers = emasc_int_layers
        if hasattr(scheduler.config, 'steps_offset') and scheduler.config.steps_offset != 1:
            deprecation_message = f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset` should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure to update the config accordingly as leaving `steps_offset` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            deprecate('steps_offset!=1', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['steps_offset'] = 1
            scheduler._internal_dict = FrozenDict(new_config)
        if hasattr(scheduler.config, 'skip_prk_steps') and scheduler.config.skip_prk_steps is False:
            deprecation_message = f'The configuration file of this scheduler: {scheduler} has not set the configuration `skip_prk_steps`. `skip_prk_steps` should be set to True in the configuration file. Please make sure to update the config accordingly as not setting `skip_prk_steps` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            deprecate('skip_prk_steps not set', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config['skip_prk_steps'] = True
            scheduler._internal_dict = FrozenDict(new_config)
        if safety_checker is not None and feature_extractor is None:
            raise ValueError("Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead.")
        is_unet_version_less_0_9_0 = hasattr(unet.config, '_diffusers_version') and version.parse(version.parse(unet.config._diffusers_version).base_version) < version.parse('0.9.0.dev0')
        is_unet_sample_size_less_64 = hasattr(unet.config, 'sample_size') and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = "The configuration file of the unet has set the default `sample_size` to smaller than 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n- CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5 \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the configuration file. Please make sure to update the config accordingly as leaving `sample_size=32` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `unet/config.json` file"
            deprecate('sample_size<64', '1.0.0', deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config['sample_size'] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        """
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError('Please install accelerate via `pip install accelerate`')
        device = torch.device(f'cuda:{gpu_id}')
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)
        if self.safety_checker is not None:
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    def _execution_device(self):
        """
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device('meta') or not hasattr(self.unet, '_hf_hook'):
            return self.device
        for module in self.unet.modules():
            if hasattr(module, '_hf_hook') and hasattr(module._hf_hook, 'execution_device') and (module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None):
        """
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            text_inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding='longest', return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (not torch.equal(text_input_ids, untruncated_ids)):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            if hasattr(self.text_encoder.config, 'use_attention_mask') and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(uncond_tokens, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
            if hasattr(self.text_encoder.config, 'use_attention_mask') and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        accepts_generator = 'generator' in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def decode_latents(self, latents, intermediate_features=None):
        latents = 1 / self.vae.config.scaling_factor * latents
        if intermediate_features:
            image = self.vae.decode(latents, intermediate_features=intermediate_features, int_layers=self.emasc_int_layers).sample
        else:
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(self, prompt, height, width, callback_steps, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible by 8 but are {height} and {width}.')
        if callback_steps is None or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.')
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.')
        elif prompt is None and prompt_embeds is None:
            raise ValueError('Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.')
        elif prompt is not None and (not isinstance(prompt, str) and (not isinstance(prompt, list))):
            raise ValueError(f'`prompt` has to be of type `str` or `list` but is {type(prompt)}')
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.')
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.')

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, return_intermediate=False):
        mask = torch.nn.functional.interpolate(mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask = mask.to(device=device, dtype=dtype)
        masked_image = masked_image.to(device=device, dtype=dtype)
        if isinstance(generator, list):
            masked_image_latents = [self.vae.encode(masked_image[i:i + 1])[0].latent_dist.sample(generator=generator[i]) for i in range(batch_size)]
            if return_intermediate:
                masked_image_intermediate_features = [self.vae.encode(masked_image[i:i + 1])[1] for i in range(batch_size)]
                masked_image_intermediate_features = [masked_image_intermediate_features[i] for i in self.emasc_int_layers]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents, masked_image_intermediate_features = self.vae.encode(masked_image)
            masked_image_latents = masked_image_latents.latent_dist.sample(generator=generator)
            if return_intermediate:
                masked_image_intermediate_features = [masked_image_intermediate_features[i] for i in self.emasc_int_layers]
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(f"The passed mask and the required batch size don't match. Masks are supposed to be duplicated to a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number of masks that you pass is divisible by the total requested batch size.")
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(f"The passed images and the required batch size don't match. Images are supposed to be duplicated to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed. Make sure the number of images that you pass is divisible by the total requested batch size.")
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        if return_intermediate:
            return (mask, masked_image_latents, masked_image_intermediate_features)
        else:
            return (mask, masked_image_latents)

    @torch.no_grad()
    def __call__(self, image: Union[torch.FloatTensor, PIL.Image.Image], mask_image: Union[torch.FloatTensor, PIL.Image.Image], pose_map: torch.FloatTensor, warped_cloth: torch.FloatTensor, prompt: Union[str, List[str]]=None, height: Optional[int]=None, width: Optional[int]=None, num_inference_steps: int=50, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, prompt_embeds: Optional[torch.FloatTensor]=None, negative_prompt_embeds: Optional[torch.FloatTensor]=None, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.FloatTensor]=None, output_type: Optional[str]='pil', return_dict: bool=True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]]=None, callback_steps: Optional[int]=1, cloth_cond_rate: float=1.0, no_pose: bool=False, cloth_input_type: str='warped'):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)
        if image is None:
            raise ValueError('`image` input cannot be undefined.')
        if mask_image is None:
            raise ValueError('`mask_image` input cannot be undefined.')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds)
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
        pose_map = torch.nn.functional.interpolate(pose_map, size=(pose_map.shape[2] // 8, pose_map.shape[3] // 8), mode='bilinear')
        if no_pose:
            pose_map = torch.zeros_like(pose_map)
        if cloth_input_type == 'warped':
            cloth_latents = self.vae.encode(warped_cloth)[0].latent_dist.sample(generator=generator)
        elif cloth_input_type == 'none':
            cloth_latents = None
        else:
            raise ValueError(f'Invalid cloth_input_type {cloth_input_type}')
        if cloth_latents is not None:
            cloth_latents = self.vae.config.scaling_factor * cloth_latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        cloth_conditioning_steps = (1 - cloth_cond_rate) * num_inference_steps
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents)
        if self.emasc:
            mask, masked_image_latents, intermediate_features = self.prepare_mask_latents(mask, masked_image, batch_size * num_images_per_prompt, height, width, prompt_embeds.dtype, device, generator, do_classifier_free_guidance, return_intermediate=True)
            intermediate_features = self.emasc(intermediate_features)
            intermediate_features = mask_features(intermediate_features, mask_image)
        else:
            mask, masked_image_latents = self.prepare_mask_latents(mask, masked_image, batch_size * num_images_per_prompt, height, width, prompt_embeds.dtype, device, generator, do_classifier_free_guidance, return_intermediate=False)
        pose_map = torch.cat([torch.zeros_like(pose_map), pose_map]) if do_classifier_free_guidance else pose_map
        if cloth_latents is not None:
            cloth_latents = torch.cat([torch.zeros_like(cloth_latents), cloth_latents]) if do_classifier_free_guidance else cloth_latents
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if i >= num_inference_steps - cloth_conditioning_steps:
                    cloth_latents = torch.zeros_like(cloth_latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if cloth_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_map.to(mask.dtype), cloth_latents.to(mask.dtype)], dim=1)
                else:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents, pose_map.to(mask.dtype)], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.to(self.vae.dtype)
                if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if self.emasc:
            image = self.decode_latents(latents, intermediate_features)
        else:
            image = self.decode_latents(latents)
        if output_type == 'pil':
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image, None)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

