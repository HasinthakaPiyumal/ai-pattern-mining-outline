# Cluster 24

class Scene(nn.Module):

    def __init__(self, config):
        self.data_root = config['data_root']
        self.scene_name = config['scene_name']
        self.ext_int_path = os.path.join(self.data_root, self.scene_name, config['ext_int_file'])
        self.bbox_path = os.path.join(self.data_root, self.scene_name, config['bbox_file'])
        self.map_path = os.path.join(self.data_root, self.scene_name, config['map_file'])
        self.pcd_path = os.path.join(self.data_root, self.scene_name, config['pcd_file'])
        self.init_img_path = os.path.join(self.data_root, self.scene_name, config['init_img_file'])
        with open(self.map_path, 'rb') as f:
            self.map_data = pickle.load(f)
        self.is_wide_angle = config['is_wide_angle']
        self.fps = config.get('fps', 20)
        self.frames = config['frames']
        self.multi_process_num = config.get('multi_process_num', 1)
        self.backup_hdri = config.get('backup_hdri', True)
        self.depth_and_occlusion = config.get('depth_and_occlusion', False)
        '\n        [static scene data] \n        '
        self.bbox_data = np.load(self.bbox_path, allow_pickle=True).item()
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        self.pcd = np.asarray(pcd.points)
        self.pcd = self.pcd[self.pcd[:, -1] > 0.5]
        all_current_vertices = []
        for k in self.bbox_data.keys():
            current_vertices = generate_vertices(self.bbox_data[k])
            all_current_vertices.append(current_vertices)
        self.all_current_vertices = np.array(all_current_vertices)
        if self.all_current_vertices.shape[0] > 0:
            self.all_current_vertices_coord = np.mean(self.all_current_vertices, axis=1)[:, :2]
        else:
            self.all_current_vertices_coord = np.zeros((0, 2))
        extrinsics = np.load(self.ext_int_path)[:, :12].reshape(-1, 3, 4)
        extrinsics = extrinsics[:, :3, :4]
        self.nerf_motion_extrinsics = extrinsics
        self.intrinsics = np.load(self.ext_int_path)[:, 12:21].reshape(-1, 3, 3)[0]
        self.focal = self.intrinsics[0, 0]
        self.height = 1280
        self.width = 1920
        if self.is_wide_angle:
            self.intrinsics[0, 2] += 1920
            self.width = 1920 * 3
        "\n        [dynamic scene data], will be updated during parsing. \n        ---\n        current_extrinsics : np.npdarray [N, 3, 4] \n            N=#frames, correspond to current_images. NeRF (RUB) convention\n\n        current_images : list of np.ndarray [H, W, 3] with len=frames\n            Show to users. NeRF's output: current_images\n\n        current_inpainted_images: list of np.ndarray [H, W, 3] with len=frames\n            Show to users. NeRF + inpaint's output: current_inpainted_images\n\n        "
        self.is_ego_motion = False
        self.add_car_all_static = True
        self.current_extrinsics = self.nerf_motion_extrinsics[3:4]
        self.current_extrinsics = self.current_extrinsics.repeat(self.frames, axis=0)
        self.removed_cars = []
        self.added_cars_dict = {}
        self.added_cars_count = 0
        self.past_operations = []
        self.all_trajectories = []
        current_time = datetime.datetime.now()
        short_scene_name = self.scene_name.lstrip('segment-')[:4]
        simulation_name = config['simulation_name']
        self.logging_name = current_time.strftime(f'{short_scene_name}_{simulation_name}_%Y_%m_%d_%H_%M_%S')
        self.save_cache = config['save_cache']
        self.cache_dir = os.path.join(config['cache_dir'], self.logging_name)
        self.output_dir = config['output_dir']
        check_and_mkdirs(self.cache_dir)
        check_and_mkdirs(self.output_dir)

    def setup_cars(self):
        """
        Call at the beginning of each interaction. 
        calculate the information of cars from original scene based on current extrinsic
        """
        original_cars_dict = {}
        name_to_bbox_car_id = {}
        bbox_car_id_to_name = {}
        mask_list = []
        mask_corners_list = []
        depth_list = []
        u_v_depth_list = []
        car_id_list = []
        for car_id in self.bbox_data.keys():
            extrinsic_for_project = transform_nerf2opencv_convention(self.current_extrinsics[0])
            u_v_depth = get_attributes_for_one_car(self.bbox_data[car_id], extrinsic_for_project, self.intrinsics)
            if u_v_depth['u'] < 0 or u_v_depth['u'] > self.width or u_v_depth['v'] < 0 or (u_v_depth['v'] > self.height):
                continue
            corners = generate_vertices(self.bbox_data[car_id])
            mask, mask_corners = get_outlines(corners, extrinsic_for_project, self.intrinsics, self.height, self.width)
            mask_list.append(mask)
            mask_corners_list.append(mask_corners)
            depth_list.append(u_v_depth['depth'])
            u_v_depth_list.append(u_v_depth)
            car_id_list.append(car_id)
        color_dict = getColorList()
        for idx_in_list, car_id in enumerate(car_id_list):
            car_name = f'original_car_{car_id}'
            name_to_bbox_car_id[car_name] = car_id
            bbox_car_id_to_name[car_id] = car_name
            original_cars_dict[car_name] = u_v_depth_list[idx_in_list]
            current_mask_corner = mask_corners_list[idx_in_list]
            color = get_color(self.current_images[0][current_mask_corner[0] + 50:current_mask_corner[1] - 50, current_mask_corner[2] + 50:current_mask_corner[3] - 50])
            color_vector = (color_dict[color][0] + color_dict[color][1]) / 2
            color_vector = np.uint8(color_vector.reshape(1, 1, 3))
            original_cars_dict[car_name]['rgb'] = cv2.cvtColor(color_vector, cv2.COLOR_HSV2RGB)
            original_cars_dict[car_name]['x'] = self.bbox_data[car_id]['cx']
            original_cars_dict[car_name]['y'] = self.bbox_data[car_id]['cy']
        self.original_cars_dict = original_cars_dict
        self.name_to_bbox_car_id = name_to_bbox_car_id
        self.bbox_car_id_to_name = bbox_car_id_to_name

    def remove_car(self, car_name):
        """
        append car_id to self.removed_cars, inpaint them later.

        car_name
        """
        self.removed_cars.append(car_name)

    def add_car(self, added_car_info):
        """
        Add a single car to self.added_cars_dict dictionary.
        added_car_id is the number of cars added so far.
        """
        added_car_info['need_placement_and_motion'] = True
        added_car_id = str(self.added_cars_count)
        car_name = f'added_car_{added_car_id}'
        self.added_cars_dict[car_name] = added_car_info
        self.added_cars_count += 1
        return car_name

    def check_added_car_static(self):
        """
        if all added cars are static, we only need to render one frame in blender
        """
        self.add_car_all_static = True
        for added_car_id, added_car_info in self.added_cars_dict.items():
            is_static = np.all(added_car_info['motion'] == added_car_info['motion'][0])
            self.add_car_all_static = self.add_car_all_static and is_static

    def clean_cache(self):
        folder_path = self.cache_dir
        shutil.rmtree(folder_path)

def generate_vertices(car):
    """Generates the vertices of a 3D box."""
    x = car['cx']
    y = car['cy']
    z = car['cz']
    length = car['length']
    width = car['width']
    height = car['height']
    heading = car['heading']
    box_center = np.array([x, y, z])
    half_dims = np.array([length / 2, width / 2, height / 2])
    relative_positions = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * half_dims
    vertices = np.asarray([rotate(pos, heading) + box_center for pos in relative_positions])
    return vertices

def transform_nerf2opencv_convention(extrinsic):
    """
    Transform and pad NeRF convention extrinsic (RUB) [3, 4] to
                      OpenCV convention extrisic (RDF) [4, 4].

    Args:
        extrinsic : np.ndarray
            shape [3, 4] in NeRF convention extrinsic (RUB)
    Returns:
        extrinsic_opencv : np.ndarray
            shape [4, 4] in OpenCV convention extrinsic (RDF)
    """
    all_ones = np.array([[0, 0, 0, 1]])
    extrinsic_opencv = np.concatenate((extrinsic, all_ones), axis=0)
    extrinsic_opencv = np.concatenate((extrinsic_opencv[:, 0:1], -extrinsic_opencv[:, 1:2], -extrinsic_opencv[:, 2:3], extrinsic_opencv[:, 3:]), axis=1)
    return extrinsic_opencv

def get_attributes_for_one_car(car, extrinsic, intrinsic):
    x = car['cx']
    y = car['cy']
    z = car['cz']
    one_point = np.array([[x, y, z]])
    all_one = np.ones((one_point.shape[0], 1))
    points = np.concatenate((one_point, all_one), axis=1).T
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]
    cam_points_without_norm = copy.copy(cam_points)
    cam_points = cam_points / cam_points[2:]
    points = (intrinsic @ cam_points).T[:, :2]
    return {'u': points[0, 0], 'v': points[0, 1], 'depth': cam_points_without_norm[-1, 0]}

def get_outlines(corners, extrinsic, intrinsic, height, width):

    def generate_convex_hull(points):
        hull = ConvexHull(points)
        return points[hull.vertices]

    def polygon_to_mask(polygon, height, width):
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon([tuple(p) for p in polygon], outline=1, fill=1)
        mask = np.array(img)
        return mask
    all_one = np.ones((corners.shape[0], 1))
    points = np.concatenate((corners, all_one), axis=1).T
    cam_points = (np.linalg.inv(extrinsic) @ points)[:3]
    cam_points = cam_points / cam_points[2:]
    points = (intrinsic @ cam_points).T[:, :2]
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    mask = np.zeros((height, width))
    points = points.astype(int)
    y_min = max(points[:, 1].min() - 50, 0)
    y_max = min(points[:, 1].max() + 50, height)
    x_min = max(points[:, 0].min() - 50, 0)
    x_max = min(points[:, 0].max() + 50, width)
    mask[y_min:y_max, x_min:x_max] = 1
    return (mask, [y_min, y_max, x_min, x_max])

def getColorList():
    dict = collections.defaultdict(list)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
    return dict

def get_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    max_num = 0
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        img, cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_num = binary[binary == 255].shape[0]
        if mask_num > max_num:
            max_num = mask_num
            color = d
    return color

def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
    if not load_imgs:
        return (poses, bds)

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    imgs = imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, -1)
    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return (poses, bds, imgs)

def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    from shutil import copy
    from subprocess import check_output
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    wd = os.getcwd()
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100.0 / r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def gen_poses(basedir, match_type, factors=None):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, match_type)
    else:
        print("Don't need to run COLMAP")
    print('Post-colmap')
    poses, pts3d, perm = load_colmap_data(basedir)
    save_poses(basedir, poses, pts3d, perm)
    if factors is not None:
        print('Factors:', factors)
        minify(basedir, factors)
    print('Done with imgs2poses')
    return True

def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))
    h, w, f = (cam.height, cam.width, cam.params[0])
    hwf = np.array([h, w, f]).reshape([3, 1])
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    poses = np.concatenate([poses[:, 0:1, :], -poses[:, 1:2, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    return (poses, pts3d, perm)

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)
    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)
    pcd = trimesh.PointCloud(pts_arr)
    pcd.export(os.path.join(basedir, 'sparse_cloud.ply'))
    os.makedirs(os.path.join(basedir, 'view_cloud'), exist_ok=True)
    for i in perm:
        vis = vis_arr[:, i]
        pts = pts_arr[vis == 1]
        pcd = trimesh.PointCloud(pts)
        pcd.export(os.path.join(basedir, 'view_cloud', '{}.ply'.format(i)))
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())
    save_arr = []
    out_vis_arr = np.zeros([vis_arr.shape[0], vis_arr.shape[1]], dtype=np.uint8)
    for j, i in enumerate(perm):
        vis = vis_arr[:, i]
        out_vis_arr[:, j] = vis.astype(np.uint8)
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = (np.percentile(zs, 0.1), np.percentile(zs, 99.9))
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    cover_ratio = out_vis_arr.astype(np.float32).sum(-1)
    cover_ratio = np.clip(cover_ratio / 10, 0.0, 1.0)
    pcd_color = np.stack([cover_ratio, 1 - cover_ratio, cover_ratio], -1)
    pcd = trimesh.PointCloud(pts_arr, pcd_color)
    pcd.export(os.path.join(basedir, 'sparse_cloud_cover_vis.ply'))
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    np.save(os.path.join(basedir, 'visibility.npy'), out_vis_arr)

def generate_rays(insert_x, insert_y, int_ext_path, nerf_exp_dir):
    near = 0.01
    far = 1000.0
    origin = np.array([insert_x, insert_y, 0.0])
    cam_meta = np.load(int_ext_path)
    extrinsic = cam_meta[:, :12].reshape(-1, 3, 4)
    translation = extrinsic[:, :3, 3].copy()
    center = np.mean(translation, axis=0)
    bias = translation - center[None]
    radius = np.linalg.norm(bias, 2, -1, False).max()
    translation = (translation - center[None]) / radius
    extrinsic[:, :, 3] = translation
    origin = (origin - center) / radius
    dy = np.linspace(0, 1, 1280)
    dx = np.linspace(0, 1, 1280 * 4)
    u, v = np.meshgrid(dx, dy)
    u, v = (u.ravel()[..., None], v.ravel()[..., None])
    rays_d = skylatlong2world(u, v)
    rays_o = np.tile(origin[None], (len(rays_d), 1))
    bounds = np.array([[near, far]]).repeat(len(rays_d), axis=0)
    np.save(os.path.join(nerf_exp_dir, 'rays_o.npy'), rays_o.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'rays_d.npy'), rays_d.astype(np.float32))
    np.save(os.path.join(nerf_exp_dir, 'bounds.npy'), bounds.astype(np.float32))

def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2
    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)
    direction = np.concatenate((-z, -x, y), axis=1)
    return direction

class DeletionAgent:

    def __init__(self, config):
        self.config = config
        self.inpaint_dir = config['inpaint_dir']
        self.video_inpaint_dir = config['video_inpaint_dir']

    def llm_finding_deletion(self, scene, message, scene_object_description):
        try:
            q0 = 'I will provide you with an operation statement and a dictionary containing information about cars in a scene. ' + ' You need to determine which car or cars should be deleted from the dictionary. '
            q1 = 'The dictionary is ' + str(scene_object_description)
            q2 = 'The keys of the dictionary are the car IDs, and the value is also a dictionary containing car detail, ' + 'including its image coordinate (u,v) in an image frame, depth, color in RGB.'
            q2 = "My statement may include information about the car's color or position. You should find out from my statement which cars should be deleted and return their car IDs"
            q3 = 'Note: (1) The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. ' + "And the larger the 'u', the more to the right; And the larger the 'v', the more to the down. " + "(2) You can judge the distance by the 'depth'. The greater the depth, the farther the distance, the smaller the depth, the closer the distance." + '(3) The description of the color may not be absolutely accurate, choose the car with the closest color.'
            q4 = "You should return a JSON dictionary, with a key: 'removed_cars'." + " 'removed_cars' contains IDs of all the cars that meet the requirements. "
            q5 = 'Note that there is no need to return any code or explanations; only provide a JSON dictionary.'
            q6 = "If the dictionary is empty, 'removed_cars' should be an empty list "
            q7 = 'The requirement is :' + message
            prompt_list = [q0, q1, q2, q3, q4, q5, q6, q7]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me to assess and maintain information in a dictionary.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Deletion Agent LLM] finding the car to delete', color='magenta', attrs=['bold'])}                     \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            deletion_car_ids = eval(answer)['removed_cars']
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {deletion_car_ids} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('[Deletion Agent LLM] finding the car to delete fails')
            return []
        return deletion_car_ids

    def llm_putting_back_deletion(self, scene, message, scene_object_description):
        try:
            deleted_object_dict = {k: v for k, v in scene_object_description.items() if k in scene.removed_cars}
            q0 = 'I will provide you with a dictionary in which each key is a vehicle id, and each value is the description of the vehicle in the image.'
            q1 = "Specifically, description of the vehicle is also a dictionary. It has keys: (1) vehicle's u in image coordinate (2) vehicle's v in image coordinate (3) vehicle color in RGB. (4) vehicle's depth from viewpoint"
            q2 = 'The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. ' + "The larger the 'u', the more to the right; And the larger the 'v', the more to the down. "
            q3 = 'I will get you a requirement, and I want you can follow this requirement and take out all the relavant vehicle ids from the dictionary.'
            q4 = f'Now the dictionary is {deleted_object_dict}, and my requirement is {message}. My requirement may contain extraneous verb descriptions or the wrong singular and plural expression, please ignore.'
            q5 = "Note that you should return a JSON dictionary, the key is 'selected_vehicle', the value includes the vehicle ids. DO NOT return anything else. I'm not asking you to write code."
            prompt_list = [q0, q1, q2, q3, q4, q5]
            result = openai.ChatCompletion.create(model='gpt-4', messages=[{'role': 'system', 'content': 'You are an assistant helping me maintain and return dictionaries.'}] + [{'role': 'user', 'content': q} for q in prompt_list])
            answer = result['choices'][0]['message']['content']
            print(f'{colored('[Deletion Agent LLM] finding the car to be put back', color='magenta', attrs=['bold'])}                      \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}')
            start = answer.index('{')
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end + 1]
            put_back_car_ids = eval(answer)['selected_vehicle']
            print(f'{colored('[Extracted Response>>>]', attrs=['bold'])} {put_back_car_ids} \n')
        except Exception as e:
            print(e)
            traceback.print_exc()
            print('[Deletion Agent LLM] finding the car to be put back fails')
        return put_back_car_ids

    def func_inpaint_scene(self, scene):
        """
        Call inpainting, store results in scene.current_inpainted_images

        if no scene.removed_cars
            just return

        """
        if len(scene.removed_cars) == 0:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} No inpainting.')
            scene.current_inpainted_images = scene.current_images
            return
        current_dir = os.getcwd()
        inpaint_input_path = os.path.join(current_dir, scene.cache_dir, 'inpaint_input')
        inpaint_output_path = os.path.join(current_dir, scene.cache_dir, 'inpaint_output')
        check_and_mkdirs(inpaint_input_path)
        check_and_mkdirs(inpaint_output_path)
        if scene.is_ego_motion is False:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is False, inpainting one frame.')
            all_mask = self.func_get_mask(scene)
            img = scene.current_images[0]
            masked_img = copy.deepcopy(img)
            if scene.is_wide_angle:
                masked_img = cv2.resize(masked_img, (1152, 256))
            else:
                masked_img = cv2.resize(masked_img, (512, 384))
            imageio.imwrite(os.path.join(inpaint_input_path, 'img.png'), masked_img.astype(np.uint8))
            imageio.imwrite(os.path.join(inpaint_input_path, 'img_mask.png'), all_mask.astype(np.uint8))
            current_dir = os.getcwd()
            os.chdir(self.inpaint_dir)
            os.system(f'python scripts/inpaint.py --indir {inpaint_input_path} --outdir {inpaint_output_path}')
            os.chdir(current_dir)
            new_img = imageio.imread(os.path.join(inpaint_output_path, 'img.png'))
            new_img = cv2.resize(new_img, (scene.width, scene.height))
            all_mask_in_ori_resolution = cv2.resize(all_mask, (scene.width, scene.height)).reshape(scene.height, scene.width, 1).repeat(3, axis=2)
            new_img = np.where(all_mask_in_ori_resolution == 0, scene.current_images[0], new_img)
            scene.current_inpainted_images = [new_img] * scene.frames
        else:
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is True, inpainting multiple frame (as video).')
            mask_list = []
            for i in range(scene.frames):
                current_frame_mask = np.zeros((scene.height, scene.width))
                for car_id in scene.bbox_data.keys():
                    if scene.bbox_car_id_to_name[car_id] in scene.removed_cars:
                        corners = generate_vertices(scene.bbox_data[car_id])
                        mask, mask_corners = get_outlines(corners, transform_nerf2opencv_convention(scene.current_extrinsics[i]), scene.intrinsics, scene.height, scene.width)
                        current_frame_mask[mask == 1] = 1
                mask_list.append(current_frame_mask)
            np.save(f'{self.video_inpaint_dir}/chatsim/masks.npy', mask_list)
            np.save(f'{self.video_inpaint_dir}/chatsim/current_images.npy', scene.current_images)
            current_dir = os.getcwd()
            os.chdir(self.video_inpaint_dir)
            os.system(f'python remove_anything_video_npy.py                         --dilate_kernel_size 15                         --lama_config lama/configs/prediction/default.yaml                         --lama_ckpt ./pretrained_models/big-lama                         --tracker_ckpt vitb_384_mae_ce_32x4_ep300                         --vi_ckpt ./pretrained_models/sttn.pth                         --mask_idx 2                         --fps 25')
            os.chdir(current_dir)
            print(f'{colored('[Inpaint]', 'green', attrs=['bold'])} Video Inpainting Done!')
            inpainted_images = np.load(f'{self.video_inpaint_dir}/chatsim/inpainted_imgs.npy', allow_pickle=True)
            scene.current_inpainted_images = [np.array(image) for image in inpainted_images]

    def func_get_mask(self, scene):
        masks = []
        extrinsic_for_project = transform_nerf2opencv_convention(scene.current_extrinsics[0])
        for car_name in scene.removed_cars:
            car_id = scene.name_to_bbox_car_id[car_name]
            corners = generate_vertices(scene.bbox_data[car_id])
            mask, _ = get_outlines(corners, extrinsic_for_project, scene.intrinsics, scene.height, scene.width)
            mask *= 255
            masks.append(mask)
        mask = np.max(np.stack(masks), axis=0)
        if scene.is_wide_angle:
            mask = cv2.resize(mask, (1152, 256))
        else:
            mask = cv2.resize(mask, (512, 384))
        return mask

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

def blending_hdr_sky(nerf_env_panorama, sky_dome_panorama, nerf_last_trans, sky_mask):
    """
    blending hdr sky dome with nerf panorama
    Args:
        nerf_env_panorama : np.ndarray
            shape [H1, W1, 3], In Linear space

        sky_dome_panorama : np.ndarray
            shape [H2, W2, 3], In Linear space

        nerf_last_trans : np.ndarray
            shape [H1, W1, 1], range (0-1)
        
    """
    H, W, _ = sky_dome_panorama.shape
    sky_mask = cv2.resize(sky_mask, (W, H))[:, :, :1]
    nerf_env_panorama = cv2.resize(nerf_env_panorama, (W, H))
    nerf_last_trans = cv2.resize(nerf_last_trans, (W, H))[:, :, np.newaxis]
    nerf_last_trans[sky_mask > 255 * 0.5] = 1
    final_hdr_sky = nerf_env_panorama + sky_dome_panorama * nerf_last_trans
    return final_hdr_sky

def srgb_gamma_correction(linear_image):
    """
    linear_image: np.ndarray
        shape: H*W*C
    """
    linear_image = np.clip(linear_image, 0, 1)
    gamma_corrected_image = np.where(linear_image <= 0.0031308, linear_image * 12.92, 1.055 * linear_image ** (1 / 2.4) - 0.055)
    gamma_corrected_image = np.clip(gamma_corrected_image, 0, 1)
    return gamma_corrected_image

class HoliCitySDRDataset(Dataset):

    def __init__(self, args, split='train'):
        self.multicrop_dir = args['multicrop_dir']
        self.skymask_dir = args['skymask_dir']
        self.skyldr_dir = args['skyldr_dir']
        self.skyhdr_dir = args['skyhdr_dir']
        selected_sample_json = args['selected_sample_json']
        view_args = args['view_setting']
        self.crop_H = view_args['camera_H'] // view_args['downsample_for_crop']
        self.crop_W = view_args['camera_W'] // view_args['downsample_for_crop']
        self.camera_vfov = np.degrees(np.arctan2(view_args['camera_H'] / 2, view_args['focal'])) * 2
        self.aspect_ratio = view_args['camera_W'] / view_args['camera_H']
        self.view_num = view_args['view_num']
        self.view_dis_deg = view_args['view_dis']
        self.sky_pano_H = args['sky_pano_H']
        self.sky_pano_W = args['sky_pano_W']
        with open(selected_sample_json, 'r') as f:
            self.select_sample = json.load(f)
        random.seed(303)
        random.shuffle(self.select_sample)
        all_sample_num = len(self.select_sample)
        train_ratio = 0.8
        self.train_file_list = self.select_sample[:int(all_sample_num * train_ratio)]
        self.val_file_list = self.select_sample[int(all_sample_num * train_ratio):]
        self.is_train = split == 'train'
        if self.is_train:
            self.file_list = self.train_file_list
        else:
            self.file_list = self.val_file_list
        self.aug_rotation = True if self.is_train else False

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sky_ldr_path = os.path.join(self.skyldr_dir, self.file_list[idx])
        sky_mask_path = os.path.join(self.skymask_dir, self.file_list[idx])
        sky_hdr_path = os.path.join(self.skyhdr_dir, self.file_list[idx].replace('.jpg', '.npz'))
        ldr_skypano = imread(sky_ldr_path) / 255
        sky_mask = imread(sky_mask_path).astype(np.float32) / 255
        sky_hdr_dict = np.load(sky_hdr_path)
        peak_vector = sky_hdr_dict['peak_vector']
        latent_vector = sky_hdr_dict['latent_vector']
        hdr_skypano = sky_hdr_dict['hdr_skypano']
        ldr_envmap = EnvironmentMap(ldr_skypano, 'skylatlong')
        hdr_envmap = EnvironmentMap(hdr_skypano, 'skylatlong')
        mask_envmap = EnvironmentMap(sky_mask, 'skylatlong')
        if self.aug_rotation:
            azimuth_deg = choice(range(0, 360, 45))
            azimuth_rad = np.radians(azimuth_deg)
            rotation_mat = rotation_matrix(azimuth=azimuth_rad, elevation=0)
            inv_rotation_mat = rotation_matrix(azimuth=-azimuth_rad, elevation=0)
        else:
            azimuth_deg = 0
        img_crops_tensor_list = []
        for i in range(self.view_num):
            azimuth_deg_i = (azimuth_deg + self.view_dis_deg[i]) % 360
            azimuth_deg_i = int(azimuth_deg_i)
            img_crop_path = os.path.join(self.multicrop_dir, str(azimuth_deg_i), self.file_list[idx])
            img_crop = imread(img_crop_path) / 255
            img_crops_tensor_list.append(totensor(img_crop))
        if self.aug_rotation:
            hdr_envmap.rotate(dcm=inv_rotation_mat)
            mask_envmap.rotate(dcm=inv_rotation_mat)
            ldr_envmap.rotate(dcm=inv_rotation_mat)
            peak_vector[:3] = (rotation_mat @ peak_vector[:3].reshape(3, 1)).flatten()
        img_crops_tensor = torch.stack(img_crops_tensor_list)
        peak_vector_tensor = totensor(peak_vector)
        latent_vector_tensor = totensor(latent_vector)
        mask_envmap_tensor = totensor(mask_envmap.data)
        hdr_envmap_tensor = totensor(hdr_envmap.data)
        ldr_envmap_tensor = totensor(ldr_envmap.data)
        return (img_crops_tensor, peak_vector_tensor, latent_vector_tensor, mask_envmap_tensor, hdr_envmap_tensor, ldr_envmap_tensor)

def totensor(x: np.ndarray):
    if len(x.shape) == 3:
        return torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
    return torch.from_numpy(x.astype(np.float32))

class HDRSkyDataset(Dataset):

    def __init__(self, args, split='train'):
        root_dir = args['root_dir']
        downsample = args['downsample']
        self.sky_H = args['image_H'] // downsample // 2
        self.sky_W = args['image_W'] // downsample
        self.root_dir = os.path.join(root_dir, split)
        self.downsample = downsample
        self.file_list = sorted(os.listdir(self.root_dir))
        self.is_train = split == 'train'
        self.env_template = EnvironmentMap(self.sky_H, 'skylatlong')
        self.center_align = args.get('center_align', False)
        self.normalize = args.get('normalize', None)
        self.aug_exposure_range = args.get('aug_exposure_range', [-2.5, 0.5])
        self.aug_temperature_range = args.get('aug_temperature_range', [1, 1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        hdr_pano = imread(file_path)
        hdr_skypano = hdr_pano[:hdr_pano.shape[0] // 2, :, :]
        hdr_skypano = cv2.resize(hdr_skypano, (self.sky_W, self.sky_H))[:, :, :3]
        if self.is_train:
            hdr_skypano = adjust_exposure(hdr_skypano, self.aug_exposure_range)
            hdr_skypano = adjust_flip(hdr_skypano)
            if not self.center_align:
                hdr_skypano = adjust_rotation(hdr_skypano)
            hdr_skypano = adjust_color_temperature(hdr_skypano, self.aug_temperature_range)
        illumination = 0.2126 * hdr_skypano[..., 0] + 0.7152 * hdr_skypano[..., 1] + 0.0722 * hdr_skypano[..., 2]
        max_index = np.argmax(illumination, axis=None)
        max_index_2d = np.unravel_index(max_index, illumination.shape)
        peak_int_v, peak_int_u = max_index_2d
        if self.center_align:
            azimuth = (self.sky_W // 2 - peak_int_u) % self.sky_W / self.sky_W * 2 * np.pi
            hdr_skypano = adjust_rotation(hdr_skypano, azimuth)
            peak_int_u = self.sky_W // 2
        peak_int = hdr_skypano[peak_int_v, peak_int_u]
        peak_dir_w_flag = self.env_template.pixel2world(peak_int_u, peak_int_v)
        peak_dir = np.array([peak_dir_w_flag[0], peak_dir_w_flag[1], peak_dir_w_flag[2]])
        ldr_skypano = srgb_gamma_correction(hdr_skypano)
        if self.normalize:
            peak_int_R = np.percentile(hdr_skypano[..., 0], self.normalize * 100)
            peak_int_G = np.percentile(hdr_skypano[..., 1], self.normalize * 100)
            peak_int_B = np.percentile(hdr_skypano[..., 2], self.normalize * 100)
            peak_int = np.array([peak_int_R, peak_int_G, peak_int_B])
            hdr_skypano = hdr_skypano / peak_int
            hdr_skypano = hdr_skypano.clip(0, 1)
        peak_vector = np.concatenate([peak_dir, peak_int], axis=-1)
        peak_vector_tensor = torch.from_numpy(peak_vector.astype(np.float32))
        hdr_skypano_tensor = torch.from_numpy(hdr_skypano.astype(np.float32)).permute(2, 0, 1)
        ldr_skypano_tensor = torch.from_numpy(ldr_skypano.astype(np.float32)).permute(2, 0, 1)
        return (ldr_skypano_tensor, hdr_skypano_tensor, peak_vector_tensor)

