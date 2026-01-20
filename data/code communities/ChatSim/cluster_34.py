# Cluster 34

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D

def loadCam(args, id, cam_info, resolution_scale):
    """
    resolution_scale is always 1.0
    """
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = (round(orig_w / (resolution_scale * args.resolution)), round(orig_h / (resolution_scale * args.resolution)))
        K = cam_info.K / (resolution_scale * args.resolution)
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        K = cam_info.K / scale
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.sky_mask is not None:
        resized_sky_mask = torch.tensor(cv2.resize(cam_info.sky_mask, resolution, interpolation=cv2.INTER_NEAREST)).to(resized_image_rgb.device)
        resized_sky_mask = resized_sky_mask == args.sky_value
    else:
        resized_sky_mask = None
    if cam_info.normal is not None:
        raise NotImplementedError('Normal maps are not supported in this version')
    else:
        resized_normal = None
    if cam_info.depth is not None:
        resized_depth = torch.tensor(cv2.resize(cam_info.depth, resolution, interpolation=cv2.INTER_NEAREST)).to(resized_image_rgb.device)
    else:
        resized_depth = None
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY, image=gt_image, gt_alpha_mask=loaded_mask, image_name=cam_info.image_name, uid=id, data_device=args.data_device, K=K, sky_mask=resized_sky_mask, normal=resized_normal, depth=resized_depth, exposure_scale=cam_info.exposure_scale)

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list

def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {'id': id, 'img_name': camera.image_name, 'width': camera.width, 'height': camera.height, 'position': pos.tolist(), 'rotation': serializable_array_2d, 'fy': fov2focal(camera.FovY, camera.height), 'fx': fov2focal(camera.FovX, camera.width)}
    return camera_entry

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def getNerfppNorm(cam_info):

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return (center.flatten(), diagonal)
    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {'translate': translate, 'radius': radius}

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def get_center_and_diag(cam_centers):
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return (center.flatten(), diagonal)

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, args):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write('Reading camera {}/{}'.format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == 'SIMPLE_PINHOLE':
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == 'PINHOLE':
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, 'Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!'
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, K=intr.params)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def qvec2rotmat(qvec):
    return np.array([[1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]], [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]], [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def readChatsimSceneInfo(args):
    """
    This is modified for ChatSim, which use points3D_waymo.ply for initialization

    points3D_waymo.ply is from recalibration with COLMAP. See data_utils/README.md for details
    """
    path = args.source_path
    images = args.images
    cams_meta_file = os.path.join(path, 'cams_meta.npy')
    ply_path = os.path.join(path, 'points3D_waymo.ply')
    images_folder = os.path.join(path, 'images')
    image_name_list = os.listdir(images_folder)
    image_file_list = [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
    image_name_list.sort()
    image_file_list.sort()
    cam_infos_unsorted = []
    cams_meta = np.load(cams_meta_file)
    for idx, cam_data in enumerate(cams_meta):
        image_path = image_file_list[idx]
        image_name = image_name_list[idx]
        image = Image.open(image_file_list[idx])
        H, W = image.size
        c2w_RUB = np.eye(4)
        c2w_RUB[:3, :] = cam_data[:12].reshape(3, 4)
        c2w_RDF = np.concatenate([c2w_RUB[:, 0:1], -c2w_RUB[:, 1:2], -c2w_RUB[:, 2:3], c2w_RUB[:, 3:4]], axis=1)
        c2w = c2w_RDF
        w2c = np.linalg.inv(c2w)
        camera_intrinsics = cam_data[12:21].reshape(3, 3)
        R = c2w[:3, :3]
        T = w2c[:3, 3]
        K = np.array([camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]])
        FoVx = 2 * np.arctan(W / (2 * camera_intrinsics[0, 0]))
        FoVy = 2 * np.arctan(H / (2 * camera_intrinsics[1, 1]))
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FoVy, FovX=FoVx, image=image, image_path=image_path, image_name=image_name, width=W, height=H, K=K)
        if args.get('load_sky_mask', False):
            sky_mask_folder = args.sky_mask_folder
            sky_mask_path = image_path.replace(os.path.basename(images_folder), sky_mask_folder)
            try:
                sky_mask = Image.open(sky_mask_path)
            except:
                sky_mask = Image.open(sky_mask_path + '.png')
            sky_mask = np.array(sky_mask)
            cam_info = cam_info._replace(sky_mask=sky_mask)
        if args.get('load_normal', False):
            normal_folder = args.normal_folder
            normal_path = image_path.replace(os.path.basename(images_folder), normal_folder).replace('.png', '.exr')
            normal = Image.open(normal_path)
            normal = np.array(normal)
            cam_info = cam_info._replace(normal=normal)
        if args.get('load_depth', False):
            depth_folder = args.depth_folder
            depth_path = image_path.replace(os.path.basename(images_folder), depth_folder).replace('.png', '.exr')
            depth = imageio.imread(depth_path)
            cam_info = cam_info._replace(depth=depth)
        if args.get('load_exposure', False):
            exposure_folder = args.exposure_folder
            exposure_path = os.path.join(image_path.split('colmap/')[0], exposure_folder, image_name + '.txt')
            with open(exposure_path, 'r') as f:
                exposure = float(f.read())
            exposure_scale = 1 + args.exposure_coefficient * exposure
            cam_info = cam_info._replace(exposure_scale=exposure_scale)
        cam_infos_unsorted.append(cam_info)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    assert os.path.exists(ply_path), 'Please run recalibration with colmap or download provided calibration files'
    try:
        pcd = fetchPlyOpen3D(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos, nerf_normalization=nerf_normalization, ply_path=ply_path)
    return scene_info

def fetchPlyOpen3D(path):
    open3d_data = open3d.io.read_point_cloud(path)
    positions = np.array(open3d_data.points)
    colors = np.array(open3d_data.colors)
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readColmapSceneInfo(args):
    path = args.source_path
    images = args.images
    try:
        cameras_extrinsic_file = os.path.join(path, f'sparse/{args.sparse_folder}', 'images.bin')
        cameras_intrinsic_file = os.path.join(path, f'sparse/{args.sparse_folder}', 'cameras.bin')
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f'sparse/{args.sparse_folder}', 'images.txt')
        cameras_intrinsic_file = os.path.join(path, f'sparse/{args.sparse_folder}', 'cameras.txt')
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = 'images' if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    if args.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % args.llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, f'sparse/{args.sparse_folder}', 'points3D.ply')
    bin_path = os.path.join(path, f'sparse/{args.sparse_folder}', 'points3D.bin')
    txt_path = os.path.join(path, f'sparse/{args.sparse_folder}', 'points3D.txt')
    if not os.path.exists(ply_path):
        print('Converting point3d.bin to .ply, will happen only the first time you open the scene.')
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos, nerf_normalization=nerf_normalization, ply_path=ply_path)
    return scene_info

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def readCamerasFromTransforms(path, transformsfile, white_background, extension='.png'):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents['camera_angle_x']
        frames = contents['frames']
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame['file_path'] + extension)
            c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert('RGBA'))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), 'RGB')
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension='.png'):
    print('Reading Training Transforms')
    train_cam_infos = readCamerasFromTransforms(path, 'transforms_train.json', white_background, extension)
    print('Reading Test Transforms')
    test_cam_infos = readCamerasFromTransforms(path, 'transforms_test.json', white_background, extension)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3d.ply')
    if not os.path.exists(ply_path):
        num_pts = 100000
        print(f'Generating random point cloud ({num_pts})...')
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos, nerf_normalization=nerf_normalization, ply_path=ply_path)
    return scene_info

def SH2RGB(sh):
    return sh * C0 + 0.5

class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

class Scene:
    gaussians: GaussianModel

    def __init__(self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, 'point_cloud'))
            else:
                self.loaded_iter = load_iteration
            print('Loading trained model at iteration {}'.format(self.loaded_iter))
        self.train_cameras = {}
        self.test_cameras = {}
        scene_info = sceneLoadTypeCallbacks[args.scene_type](args)
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, 'input.ply'), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as file:
                json.dump(json_cams, file)
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)
        self.cameras_extent = scene_info.nerf_normalization['radius']
        for resolution_scale in resolution_scales:
            print('Loading Training Cameras')
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print('Loading Test Cameras')
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'point_cloud.ply'))
            sky_weigth_path = os.path.join(self.model_path, 'point_cloud', 'iteration_' + str(self.loaded_iter), 'sky_weight.pth')
            if self.gaussians.sky_model is not None and os.path.exists(sky_weigth_path):
                self.gaussians.sky_model.restore(torch.load(sky_weigth_path))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, 'point_cloud/iteration_{}'.format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, 'point_cloud.ply'))
        if self.gaussians.sky_model is not None:
            torch.save(self.gaussians.sky_model.capture(), os.path.join(point_cloud_path, 'sky_weight.pth'))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split('_')[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class Camera(nn.Module):

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda', K=None, sky_mask=None, normal=None, depth=None, exposure_scale=None):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.K = K
        self.sky_mask = sky_mask
        self.normal = normal
        self.depth = depth
        self.exposure_scale = exposure_scale
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f'[Warning] Custom device {data_device} failed, fallback to default cuda device')
            self.data_device = torch.device('cuda')
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixFromK(self.znear, self.zfar, self.image_width, self.image_height, self.K).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def getProjectionMatrixFromK(znear, zfar, W, H, K):
    """
    Args:
        K is np.ndarray including [fx, fy, cx, cy]
    """
    fx, fy, cx, cy = K.tolist()
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1
    P[1, 2] = 2 * (cy / H) - 1
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

