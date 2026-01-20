# Cluster 27

def training(args):
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        tb_writer = None
        print('Tensorboard not available: not logging progress')
    first_iter = 0
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians)
    gaussians.training_setup(args)
    if args.start_checkpoint:
        model_params, first_iter = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, args)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, args.iterations), desc='Training progress')
    first_iter += 1
    for iteration in range(first_iter, args.iterations + 1):
        if args.gui:
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, args.convert_SHs_python, args.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, args, background, scaling_modifer)['render']
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, args.source_path)
                    if do_training and (iteration < int(args.iterations) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        bg = torch.rand(3, device='cuda') if args.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, args, bg, exposure_scale=viewpoint_cam.exposure_scale)
        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])
        if args.render_depth:
            depth = render_pkg['depth']
        if args.render_opacity:
            opacity = render_pkg['opacity']
        loss_dict = {}
        gt_image = viewpoint_cam.original_image.cuda()
        loss_l1 = l1_loss(image, gt_image)
        loss_dict['l1_loss'] = loss_l1.item()
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss_dict['ssim_loss'] = loss_ssim.item()
        loss = (1.0 - args.lambda_dssim) * loss_l1 + args.lambda_dssim * loss_ssim
        if args.get('lambda_opacity', 0.0) > 0.0:
            sky_mask = viewpoint_cam.sky_mask.cuda()
            opacity_mask = ~sky_mask
            opacity_mask = opacity_mask.float().unsqueeze(0)
            opacity = opacity.clamp(1e-06, 1.0 - 1e-06)
            loss_opacity = -(opacity_mask * torch.log(opacity) + (1 - opacity_mask) * torch.log(1 - opacity)).mean()
            loss += args.lambda_opacity * loss_opacity
            loss_dict['opacity_loss'] = loss_opacity.item()
        if args.get('lambda_depth', 0.0) > 0.0:
            depth_mask = depth_mask > 0
            loss_depth = (torch.abs(depth - viewpoint_cam.depth.to('cuda')) * depth_mask).mean()
            loss += args.lambda_depth * loss_depth
            loss_dict['depth_loss'] = loss_depth.item()
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix_dict = {'EMA Loss': f'{ema_loss_for_log:.{3}f}'}
                for key, value in loss_dict.items():
                    postfix_dict[key] = f'{value:.{3}f}'
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == args.iterations:
                progress_bar.close()
            training_report(tb_writer, iteration, loss_dict, iter_start.elapsed_time(iter_end), args.testing_iterations, scene, render, (args, background))
            if iteration in args.saving_iterations:
                print('\n[ITER {}] Saving Gaussians'.format(iteration))
                scene.save(iteration)
            if iteration < args.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])
                if iteration > args.densify_from_iter and iteration % args.densification_interval == 0:
                    size_threshold = 20 if iteration > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % args.opacity_reset_interval == 0 or (args.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration < args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in args.checkpoint_iterations:
                print('\n[ITER {}] Saving Checkpoint'.format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + '/chkpnt' + str(iteration) + '.pth')

