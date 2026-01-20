# Cluster 26

def render_sets(args, iteration: int, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        if not skip_test:
            render_set(args, 'test', scene.loaded_iter, scene.getTestCameras(), gaussians, background)
        if not skip_train:
            render_set(args, 'train', scene.loaded_iter, scene.getTrainCameras(), gaussians, background)

def render_set(args, name, iteration, views, gaussians, background):
    model_path = args.model_path
    render_path = os.path.join(model_path, name, 'ours_{}'.format(iteration), 'renders')
    gts_path = os.path.join(model_path, name, 'ours_{}'.format(iteration), 'gt')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if args.render_depth:
        depth_path = os.path.join(model_path, name, 'ours_{}'.format(iteration), 'depth')
        makedirs(depth_path, exist_ok=True)
    if args.render_opacity:
        opacity_path = os.path.join(model_path, name, 'ours_{}'.format(iteration), 'opacity')
        makedirs(opacity_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc='Rendering progress')):
        render_pkg = render(view, gaussians, args, background, exposure_scale=view.exposure_scale)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(render_pkg['render'], os.path.join(render_path, '{0:05d}'.format(idx) + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + '.png'))

def create_view_cameras(camera_extrinsics, camera_intrinsics, H, W):
    """
    We will transform the camera extrinsics and intrinsics to scene.cameras.Camera objects.

    Note 1) camera extrinsics are RUB, but gaussians splatting requires COLMAP convention (RDF)
    Note 2) R is c2w, T is w2c. We need to inverse the camera_extrinsics to get T.

    Args:
        camera_extrinsics: [N_frames, 3, 4], c2w
        camera_intrinsics: [3, 3]
        H: height of the image
        W: width of the image
    """
    frames_num = camera_extrinsics.shape[0]
    camera_extrinsics = np.concatenate([camera_extrinsics[:, :, 0:1], -camera_extrinsics[:, :, 1:2], -camera_extrinsics[:, :, 2:3], camera_extrinsics[:, :, 3:4]], axis=2)
    view_cameras = []
    for i in tqdm(range(frames_num)):
        c2w = np.eye(4)
        c2w[:3] = camera_extrinsics[i]
        w2c = np.linalg.inv(c2w)
        R = c2w[:3, :3]
        T = w2c[:3, 3]
        K = np.array([camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]])
        FoVx = 2 * np.arctan(W / (2 * camera_intrinsics[0, 0]))
        FoVy = 2 * np.arctan(H / (2 * camera_intrinsics[1, 1]))
        image = torch.zeros((3, H, W), dtype=torch.float32)
        image_name = f'image_{i:03d}'
        uid = i
        camera = Camera(colmap_id=uid, R=R, T=T, FoVx=FoVx, FoVy=FoVy, image=image, gt_alpha_mask=None, image_name=image_name, uid=uid, K=K)
        view_cameras.append(camera)
    return view_cameras

